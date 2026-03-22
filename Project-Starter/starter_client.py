"""
PriceScout MCP Client - Interactive chatbot that talks to multiple MCP servers,
leverages Anthropic's Claude LLM for natural-language processing, and extracts
structured pricing data from the LLM responses.

The module is organized around four public classes:

* ``Configuration`` - Loads environment variables and validates the
  required `ANTHROPIC_API_KEY`.
* ``Server`` - Wraps a single MCP server instance (via :class:`mcp.ClientSession`)
  and exposes helper methods for tool discovery, execution and cleanup.
* ``DataExtractor`` - Uses Claude to parse LLM output into a JSON schema,
  then stores the data in an SQLite-backed MCP server.
* ``ChatSession`` - Orchestrates the end-to-end flow: user → LLM → tools →
  extraction → persistence.  It also implements a simple CLI for
  querying and displaying stored pricing plans.

Running this file directly (`python chat.py`) starts the interactive
chatbot loop; it will automatically discover all configured MCP servers,
register their tools, and expose them to the user.
"""

import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, List, Dict, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import re
import ast

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --------------------------------------------------------------------------- #
# Configuration & global objects
# --------------------------------------------------------------------------- #

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ANTHROPIC_BASE_MODEL = os.environ.get(
    "ANTHROPIC_BASE_MODEL", "claude-sonnet-4-5-20250929"
)
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://claude.vocareum.com")


class ToolDefinition(TypedDict):
    """Typed dictionary representing a tool's metadata."""

    name: str
    description: str
    input_schema: dict


# --------------------------------------------------------------------------- #
# Helper classes
# --------------------------------------------------------------------------- #


class Configuration:
    """
    Load and validate configuration for the MCP client.

    The class is responsible for:

    * Loading environment variables from ``.env`` (via :func:`dotenv.load_dotenv`).
    * Reading a JSON configuration file that describes the MCP servers to
      connect to.
    * Exposing the Anthropic API key as a property, raising an informative
      exception if it is missing.

    The configuration file must contain a top-level ``mcpServers`` object,
    e.g.::

        {
            "mcpServers": {
                "llm_inference": {
                    "command": "npx",
                    "args": ["--port", "8000"],
                    "env": {"VAR1": "value"}
                }
            }
        }

    Attributes
    ----------
    api_key : str | None
        The Anthropic API key read from the environment.
    """

    def __init__(self) -> None:
        """Initialise configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from a ``.env`` file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """
        Load server configuration from JSON file.

        Parameters
        ----------
        file_path : str | pathlib.Path
            Path to the JSON configuration file.

        Returns
        -------
        dict[str, Any]
            Dictionary containing server configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        json.JSONDecodeError
            If the file contains invalid JSON.
        ValueError
            If the required ``mcpServers`` key is missing.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "mcpServers" not in config:
                raise ValueError("Configuration file is missing 'mcpServers' field")

            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file: {e}"
            )  # type: ignore

    @property
    def anthropic_api_key(self) -> str:
        """
        Return the Anthropic API key.

        Raises
        ------
        ValueError
            If ``ANTHROPIC_API_KEY`` is not set in the environment.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Wraps a single MCP server instance and exposes helper methods."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """
        Create a new :class:`Server` instance.

        Parameters
        ----------
        name : str
            Human-readable identifier for the server.
        config : dict[str, Any]
            Dictionary containing configuration options such as command,
            arguments, and environment variables.
        """
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """
        Initialise the MCP server connection.

        The method resolves the command to run (``npx`` or a custom shell
        script), creates :class:`StdioServerParameters`, and then opens an
        asynchronous stdio transport via :func:`stdio_client`.  A
        :class:`ClientSession` is created on top of that transport.

        Raises
        ------
        ValueError
            If the command cannot be resolved.
        RuntimeError
            If any step in the connection chain fails; the error is logged,
            resources are cleaned up, and the exception is re-raised.
        """
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]} if self.config.get("env") else None
            ),
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """
        Retrieve the list of tools exposed by this MCP server.

        Returns
        -------
        List[ToolDefinition]
            A list of dictionaries containing ``name``, ``description`` and
            ``input_schema`` for each tool.

        Raises
        ------
        RuntimeError
            If the server has not been initialised.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} is not initialized")

        tools_response = await self.session.list_tools()

        tools: List[ToolDefinition] = []
        for tool in tools_response.tools:
            schema = tool.inputSchema or {}
            if "type" not in schema:
                schema = {**schema, "type": "object", "properties": {}}

            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,  # type: ignore
                    "input_schema": schema,
                }
            )
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 3,
        delay: float = 1.0,
    ) -> Any:
        """
        Execute a tool with retry logic.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute.
        arguments : dict[str, Any]
            Arguments for the tool.
        retries : int, default 3
            Number of retry attempts before giving up.
        delay : float, default 1.0
            Seconds to wait between retries.

        Returns
        -------
        Any
            The raw result returned by the MCP server.

        Raises
        ------
        RuntimeError
            If the server has not been initialised.
        Exception
            Propagated after exhausting all retry attempts.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} is not initialized")

        for attempt in range(retries + 1):
            try:
                logging.info(f"Executing {tool_name}...")
                result = await asyncio.wait_for(
                    self.session.call_tool(name=tool_name, arguments=arguments),
                    timeout=60,
                )
                return result
            except Exception as e:
                if attempt < retries:
                    logging.warning(
                        f"Tool execution failed (attempt "
                        f"{attempt + 1}/{retries + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logging.error(
                        f"Tool execution failed after {retries + 1} attempts: {e}"
                    )
                    raise

    async def cleanup(self) -> None:
        """
        Close all resources associated with this server.

        The method is idempotent; it can be called multiple times
        without side effects.  Any exception during the close process
        is logged but not re-raised.
        """
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""

    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        """
        Initialise the extractor.

        Parameters
        ----------
        sqlite_server : Server
            Instance used for executing SQL queries against an SQLite MCP server.
        anthropic_client : Anthropic
            Client used to communicate with Claude for structured extraction.
        """
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client

    async def setup_data_tables(self) -> None:
        """Create the ``pricing_plans`` table if it does not already exist."""
        try:
            await self.sqlite_server.execute_tool(
                "write_query",
                {
                    "query": """
                        CREATE TABLE IF NOT EXISTS pricing_plans (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            company_name TEXT NOT NULL,
                            plan_name TEXT NOT NULL,
                            input_tokens REAL,
                            output_tokens REAL,
                            currency TEXT DEFAULT 'USD',
                            billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                            features TEXT,        -- JSON array
                            limitations TEXT,
                            source_query TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                },
            )
            logging.info("✓ Data extraction tables initialized")
        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Ask Claude to return a plain text representation of the pricing data.

        Parameters
        ----------
        prompt : str
            Prompt sent to the LLM for extraction.

        Returns
        -------
        str
            The raw text returned by Claude.  If an error occurs,
            a JSON string containing ``{"error": "extraction failed"}`` is
            returned instead.
        """
        try:
            response = self.anthropic.messages.create(
                max_tokens=1024,
                model=ANTHROPIC_BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )

            text_content = ""
            for content in response.content:
                if content.type == "text":
                    text_content += content.text

            return text_content.strip()
        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'

    async def extract_and_store_data(
        self,
        user_query: str,
        llm_response: str,
        source_url: str | None = None,  # type: ignore
    ) -> None:
        """
        Parse the LLM response into JSON and persist it in SQLite.

        Parameters
        ----------
        user_query : str
            Original query submitted by the user.
        llm_response : str
            Raw text returned by Claude.
        source_url : str | None, optional
            URL of the source document (if any).

        Returns
        -------
        None
            The method writes to the ``pricing_plans`` table and logs
            progress; it does not return a value.

        Notes
        -----
        * The prompt is constructed so that Claude returns a JSON object
          with a top-level ``plans`` array.  If the response contains no
          plans, the method simply logs the fact and exits.
        * All string values are escaped for SQL to avoid injection issues.
        """
        try:
            extraction_prompt = f"""
                Analyze this text and extract pricing information in JSON format:

                Text: {llm_response}

                Extract pricing plans with this structure:
                {{
                    "company_name": "company name",
                    "plans": [
                        {{
                            "plan_name": "plan name",
                            "input_tokens": number or null,
                            "output_tokens": number or null,
                            "currency": "USD",
                            "billing_period": "monthly/yearly/one-time",
                            "features": ["feature1", "feature2"],
                            "limitations": "any limitations mentioned",
                            "query": "the user's query"
                        }}
                    ]
                }}

                Return only valid JSON, no other text. Do not return your response
                enclosed in ```json```.
            """

            extraction_response = await self._get_structured_extraction(
                extraction_prompt
            )
            extraction_response = (
                extraction_response.replace("```json\n", "").replace("```", "").strip()
            )

            if not extraction_response or "no pricing" in extraction_response.lower():
                logger.info("No pricing data found in response")
                return

            pricing_data = json.loads(extraction_response)

            if not pricing_data.get("plans"):
                logger.info("No pricing plans extracted")
                return

            for plan in pricing_data.get("plans", []):
                company = pricing_data.get("company_name", "Unknown").replace("'", "''")
                plan_name = plan.get("plan_name", "Unknown Plan").replace("'", "''")
                limitations = str(plan.get("limitations", "")).replace("'", "''")
                query_escaped = user_query.replace("'", "''")
                features = json.dumps(plan.get("features", [])).replace("'", "''")

                await self.sqlite_server.execute_tool(
                    "write_query",
                    {
                        "query": f"""
                            INSERT INTO pricing_plans (
                                company_name,
                                plan_name,
                                input_tokens,
                                output_tokens,
                                currency,
                                billing_period,
                                features,
                                limitations,
                                source_query
                            )
                            VALUES (
                                '{company}',
                                '{plan_name}',
                                '{plan.get("input_tokens", 0)}',
                                '{plan.get("output_tokens", 0)}',
                                '{plan.get("currency", "USD")}',
                                '{plan.get("billing_period", "unknown")}',
                                '{features}',
                                '{limitations}',
                                '{query_escaped}'
                            )
                        """
                    },
                )

            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans")
        except json.JSONDecodeError as e:
            logger.debug(f"Could not parse pricing data as JSON: {e}")
        except Exception as e:
            logger.debug(f"Could not extract pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        """
        Initialise chat session with available servers and Anthropic client.

        Parameters
        ----------
        servers : List[Server]
            MCP server instances to connect to.
        api_key : str
            API key for the Anthropic LLM service.
        """
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(base_url=ANTHROPIC_BASE_URL, api_key=api_key)
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Close all servers in reverse order of initialization."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_query(self, query: str) -> None:
        """
        Send a user query to Claude and handle the resulting tool calls.

        Parameters
        ----------
        query : str
            The natural-language question or request from the user.
        """
        messages = [{"role": "user", "content": query}]
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model=ANTHROPIC_BASE_MODEL,
            tools=self.available_tools,  # type: ignore
            messages=messages,  # type: ignore
        )

        full_response = ""
        source_url = None
        used_web_search = False

        i = 0
        max_iterations = 5
        process_query = True

        while process_query and i < max_iterations:
            if response.content is None:
                logger.error("API returned None content")
                process_query = False
                break

            assistant_content = list(response.content)
            tool_results: list[dict[str, Any]] = []

            for content in response.content:
                if content.type == "text":
                    full_response = content.text + "\n"
                elif content.type == "tool_use":
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name

                    if tool_name in self.tool_to_server:
                        server_name = self.tool_to_server[tool_name]
                        for _server in self.servers:
                            if _server.name == server_name:
                                tool_result = await _server.execute_tool(
                                    tool_name, tool_args  # type: ignore
                                )

                                result_text = ""
                                if tool_result.content and len(tool_result.content) > 0:
                                    result_text = (
                                        tool_result.content[0].text
                                        if hasattr(tool_result.content[0], "text")
                                        else str(tool_result.content[0])
                                    )
                                    if len(result_text) > 6000:
                                        result_text = (
                                            result_text[:6000]
                                            + "\n\n[Truncated for context management]"
                                        )
                                else:
                                    result_text = "Tool returned no content"

                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": result_text,
                                    }
                                )
                                break
                    else:
                        logger.warning(f"Unknown tool: {tool_name}")

            if response.stop_reason == "end_turn":
                process_query = False
            elif tool_results:
                messages.append(
                    {"role": "assistant", "content": assistant_content}  # type: ignore
                )  # type: ignore
                messages.append({"role": "user", "content": tool_results})  # type: ignore

                new_response = self.anthropic.messages.create(
                    max_tokens=2024,
                    model=ANTHROPIC_BASE_MODEL,
                    tools=self.available_tools,  # type: ignore
                    messages=messages,  # type: ignore
                )
                response = new_response

                if new_response.content:
                    logger.info(
                        f"Response stop_reason: {new_response.stop_reason}, "
                        f"content types: {[c.type for c in new_response.content]}"
                    )
                else:
                    logger.error(
                        f"API returned None content. Stop reason: {new_response.stop_reason}"
                    )
                    process_query = False
            else:
                process_query = False

            i += 1

        if full_response.strip():
            print(full_response.strip())

        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(
                query, full_response.strip(), source_url  # type: ignore
            )

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """
        Extract the first URL found in a block of text.

        Parameters
        ----------
        result_text : str
            Text content returned by a tool.

        Returns
        -------
        str | None
            The first matching URL or ``None`` if no URL is present.
        """
        url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break
                elif query.lower() == "show data":
                    await self.show_stored_data()
                    continue

                await self.process_query(query)
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self) -> None:
        """Display the most recent pricing plans stored in SQLite."""
        if not self.sqlite_server:
            logger.info("No database available")
            return

        try:
            pricing = await self.sqlite_server.execute_tool(
                "read_query",
                {
                    "query": (
                        "SELECT company_name, plan_name, input_tokens, "
                        "output_tokens, currency FROM pricing_plans "
                        "ORDER BY created_at DESC LIMIT 5"
                    )
                },
            )

            print("\nRecently Stored Data:")
            print("=" * 50)
            print("\nPricing Plans:")

            if pricing.content and len(pricing.content) > 0:
                result_text = (
                    pricing.content[0].text
                    if hasattr(pricing.content[0], "text")
                    else str(pricing.content[0])
                )

                if not result_text or result_text.strip() in ("[]", ""):
                    print("  No pricing data found")
                    return

                try:
                    plans = json.loads(result_text)
                    for plan in plans:
                        print(
                            f"  • {plan.get('company_name', 'N/A')}: "
                            f"{plan.get('plan_name', 'N/A')} - Input: ${plan.get('input_tokens', 'N/A')}, "
                            f"Output: ${plan.get('output_tokens', 'N/A')}"
                        )
                except json.JSONDecodeError:
                    try:
                        plans = ast.literal_eval(result_text)
                        for plan in plans:
                            print(
                                f"  • {plan.get('company_name', 'N/A')}: "
                                f"{plan.get('plan_name', 'N/A')} - Input: ${plan.get('input_tokens', 'N/A')}, "
                                f"Output: ${plan.get('output_tokens', 'N/A')}"
                            )
                    except (ValueError, SyntaxError):
                        print(result_text)
            else:
                print("  No pricing data found")

            print("=" * 50)

        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main entry point for the chat session."""
        try:
            # Initialise all servers
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            # Discover tools from each server
            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")

            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            # Start the interactive loop
            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialise configuration and start the chat session."""
    config = Configuration()

    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"

    server_config = config.load_config(config_file)

    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
