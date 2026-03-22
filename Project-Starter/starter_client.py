"""
PriceScout MCP Client

This module implements an interactive chatbot that communicates with multiple
MCP (Machine-Learning Control Plane) servers, uses Anthropic's Claude LLM to
interpret natural language queries, and persists extracted pricing data in a
SQLite database.

Public classes:

* ``Configuration`` - Loads and validates the JSON configuration file.
* ``Server`` - Wraps an MCP server instance, exposing helper methods for
    tool discovery, execution, and cleanup.
* ``DataExtractor`` - Parses LLM output into structured data and stores it
    in SQLite via the MCP write_query tool.
* ``ChatSession`` - Orchestrates user input → LLM → tools → extraction →
    persistence. It also provides a simple CLI for querying stored pricing plans.

Running this file directly (`python chat.py`) starts an interactive loop that
automatically discovers all configured MCP servers, registers their tools,
and exposes them to the user.
"""

import os
import json
import ast
import asyncio
import logging
from datetime import timedelta
from typing import Any, TypedDict
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Type definition for tool info
class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    @staticmethod
    def load_config(file_path: str) -> dict:
        """
        Load a JSON configuration file and validate its contents.

        The configuration must contain an ``mcpServers`` key mapping server names
        to their respective command, arguments, and environment variables.
        Any deviation from this structure will raise a descriptive exception.

        Parameters
        ----------
        file_path : str
            Path to the JSON configuration file.

        Returns
        -------
        dict
            The parsed and validated configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file contains invalid JSON or is missing required keys.
        """
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

        # Validate that mcpServers key exists
        if "mcpServers" not in config:
            raise ValueError("Configuration must contain 'mcpServers' key")

        return config


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict):
        """
        Create a new :class:`Server` instance.

        Parameters
        ----------
        name : str
            Identifier for the server (e.g., ``'llm_inference'`` or ``'sqlite'``).
        config : dict
            Server configuration containing command, args, and optional env.
        """
        self.name = name
        self.config = config
        self.session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None

    async def initialize(self, exit_stack: AsyncExitStack) -> "Server":
        """
        Establish a connection to the MCP server.

        The method creates a stdio client based on the provided command and
        environment variables, then initializes an :class:`mcp.ClientSession`
        that will be used for all subsequent tool interactions.

        Parameters
        ----------
        exit_stack : AsyncExitStack
            Context manager stack used to automatically close streams when
            the application exits.

        Returns
        -------
        Server
            The initialized server instance (self) for method chaining.
        """
        # Get command from config
        command = self.config.get("command")

        # Create server parameters with command, args, and environment
        server_params = StdioServerParameters(
            command=command,  # type: ignore
            args=self.config.get("args", []),
            env=(
                {**os.environ, **self.config.get("env", {})}
                if self.config.get("env")
                else None
            ),
        )

        # Create stdio client connection
        self._read_stream, self._write_stream = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        # Create and initialize session
        self.session = await exit_stack.enter_async_context(
            ClientSession(self._read_stream, self._write_stream)
        )

        await self.session.initialize()
        logger.info(f"✓ Server '{self.name}' initialized")

        return self

    async def list_tools(self) -> list[ToolDefinition]:
        """
        Retrieve the list of tools exposed by this MCP server.

        The method queries the underlying :class:`mcp.ClientSession` and
        normalizes each tool into a :class:`ToolDefinition` dictionary.

        Returns
        -------
        list[ToolDefinition]
            A list containing name, description, and input schema for each tool.
        """
        # Check if session exists
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        # Call the session to get tools
        tools_response = await self.session.list_tools()

        # Format the response into our ToolDefinition format
        tools = []
        for tool in tools_response.tools:
            tool_def: ToolDefinition = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            tools.append(tool_def)

        return tools

    async def execute_tool(
        self, tool_name: str, arguments: dict, max_retries: int = 3
    ) -> Any:
        """
        Execute a named tool on the MCP server with retry logic.

        The method attempts to call the specified tool up to ``max_retries``
        times, waiting briefly between attempts. A timeout of 60 seconds is
        applied to each individual call.

        Parameters
        ----------
        tool_name : str
            Name of the tool to invoke.
        arguments : dict
            Dictionary of arguments to pass to the tool.
        max_retries : int, optional
            Maximum number of retry attempts (default: 3).

        Returns
        -------
        Any
            The raw result returned by the MCP server.

        Raises
        ------
        RuntimeError
            If the tool fails after all retry attempts or if the server is not
            initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        last_error = None

        # Retry loop
        for attempt in range(max_retries):
            try:
                logging.info(f"Executing {tool_name}...")

                # Call the tool with 60-second timeout
                result = await self.session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(seconds=60),
                )

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {tool_name}: {e}"
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief pause before retry

        raise RuntimeError(
            f"Tool {tool_name} failed after {max_retries} attempts: {last_error}"
        )


class DataExtractor:
    """Handles extraction and storage of pricing data to SQLite."""

    def __init__(self, sqlite_server: Server):
        """
        Create a new :class:`DataExtractor` instance.

        Parameters
        ----------
        sqlite_server : Server
            Reference to the MCP server that exposes ``write_query`` for
            interacting with the SQLite database.
        """
        self.sqlite_server = sqlite_server

    async def ensure_table_exists(self):
        """Create the pricing_plans table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pricing_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            plan_name TEXT,
            input_tokens TEXT,
            output_tokens TEXT,
            currency TEXT,
            billing_period TEXT,
            features TEXT,
            limitations TEXT,
            source_query TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            await self.sqlite_server.execute_tool(
                "write_query", {"query": create_table_sql}
            )
        except Exception as e:
            logger.error(f"Error creating table: {e}")

    async def extract_and_store_data(self, pricing_data: dict, user_query: str) -> str:
        """
        Insert pricing plans into the SQLite database.

        The method first ensures that the ``pricing_plans`` table exists,
        then iterates over each plan in ``pricing_data['plans']``, inserting
        a row for each. Any insertion errors are logged but do not abort
        the entire operation.

        Parameters
        ----------
        pricing_data : dict
            Dictionary containing ``company_name`` and a list of plan
            dictionaries.
        user_query : str
            The original query that prompted this extraction, stored as
            ``source_query``.

        Returns
        -------
        str
            A status message indicating how many plans were inserted.
        """
        await self.ensure_table_exists()

        inserted = 0

        # Iterate through the plans in pricing_data
        for plan in pricing_data.get("plans", []):
            try:
                # Execute the write_query to insert data
                await self.sqlite_server.execute_tool(
                    "write_query",
                    {
                        "query": f"""
                    INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
                    VALUES (
                        '{pricing_data.get("company_name", "Unknown")}',
                        '{plan.get("plan_name", "Unknown Plan")}',
                        '{plan.get("input_tokens", 0)}',
                        '{plan.get("output_tokens", 0)}',
                        '{plan.get("currency", "USD")}',
                        '{plan.get("billing_period", "unknown")}',
                        '{json.dumps(plan.get("features", []))}',
                        '{plan.get("limitations", "")}',
                        '{user_query}'
                    )
                    """
                    },
                )
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting plan {plan.get('plan_name')}: {e}")

        return f"Inserted {inserted} pricing plans into database"


class ChatSession:
    """Manages interactive chat session with LLM and MCP tools."""

    def __init__(self, servers: dict[str, Server], anthropic_client: Anthropic):
        """
        Create a new :class:`ChatSession`.

        Parameters
        ----------
        servers : dict[str, Server]
            Mapping of server names to their corresponding :class:`Server`
            instances.
        anthropic_client : Anthropic
            Instance of the Anthropic API client used for LLM interactions.
        """
        self.servers = servers
        self.anthropic = anthropic_client
        self.messages: list[dict] = []
        self.all_tools: list[dict] = []
        self.tool_to_server: dict[str, Server] = {}

        # Get sqlite server for data extraction
        self.sqlite_server = servers.get("sqlite")
        if self.sqlite_server:
            self.data_extractor = DataExtractor(self.sqlite_server)
        else:
            self.data_extractor = None

    async def initialize_tools(self):
        """Gather tools from all servers and build mapping."""
        # Ensure pricing_plans table exists at startup
        if self.sqlite_server:
            try:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT,
                    plan_name TEXT,
                    input_tokens TEXT,
                    output_tokens TEXT,
                    currency TEXT,
                    billing_period TEXT,
                    features TEXT,
                    limitations TEXT,
                    source_query TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                await self.sqlite_server.execute_tool(
                    "write_query", {"query": create_table_sql}
                )
                logger.info("Ensured pricing_plans table exists")
            except Exception as e:
                logger.error(f"Error creating pricing_plans table: {e}")

        for server_name, server in self.servers.items():
            try:
                tools = await server.list_tools()
                for tool in tools:
                    tool_name = tool["name"]
                    # Ensure input_schema has required 'type' field
                    input_schema = tool.get("input_schema", {})
                    if not input_schema or "type" not in input_schema:
                        input_schema = {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }
                    self.all_tools.append(
                        {
                            "name": tool_name,
                            "description": tool.get("description", ""),
                            "input_schema": input_schema,
                        }
                    )
                    self.tool_to_server[tool_name] = server
                    logger.info(f"Registered tool: {tool_name} from {server_name}")
            except Exception as e:
                logger.error(f"Error listing tools from {server_name}: {e}")

    async def process_query(self, user_input: str) -> str:
        """
        Process a user query through the agentic loop.

        The method sends the user's message to Claude, handles any tool calls
        returned by the model, executes those tools via the appropriate MCP
        server, and finally returns the LLM's textual response.  Tool results
        are fed back into the conversation so that subsequent LLM turns can
        incorporate them.

        Parameters
        ----------
        user_input : str
            The raw text entered by the user.

        Returns
        -------
        str
            The final, fully rendered answer from Claude.
        """
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})

        # System prompt for the pricing analyst
        system_prompt = """
            You are an LLM pricing-intelligence assistant.  
            Your mission is twofold:

            1. **Persist every discovered pricing plan** in the SQLite table `pricing_plans` *before* you give any analysis.
            2. After all inserts succeed, produce a concise, data-driven summary of the competitive landscape.

            ---

            ### Table schema
            ```sql
            CREATE TABLE IF NOT EXISTS pricing_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name   TEXT,
                plan_name      TEXT,
                input_tokens   TEXT,
                output_tokens  TEXT,
                currency       TEXT,
                billing_period TEXT,
                features       TEXT,          -- JSON string or NULL
                limitations    TEXT,          -- plain text or NULL
                source_query   TEXT,          -- the original user query
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            ```

            ### Critical requirement  
            For **every** pricing plan you find in the scraped data, **you must call the `write_query` tool exactly once** with an INSERT statement that populates *all* columns above.  
            If a field is missing in the scraped JSON, insert `NULL` (or an empty string for text fields).  
            Use the original user query as `source_query`.

            ### Workflow
            1. **Retrieve data**  
                Call the `extract_scraped_info` tool to get a JSON blob that contains provider metadata and raw content.

            2. **Parse & insert**  
                For each plan discovered in that JSON:
                ```sql
                INSERT INTO pricing_plans
                    (company_name, plan_name, input_tokens, output_tokens,
                    currency, billing_period, features, limitations, source_query)
                VALUES
                    ('{company_name}', '{plan_name}', '{input_tokens}', '{output_tokens}',
                    '{currency}', '{billing_period}', '{features}', '{limitations}', '{source_query}');
                ```
            * Replace placeholders with the actual values from the JSON.  
            * Escape single quotes in any string value (e.g., `O'Reilly` → `O''Reilly`).  
            * Call `write_query` with this query.

            3. **Analyze**  
                After all inserts are done, output a brief analysis of the pricing landscape (e.g., best-value plans, price trends, gaps).

            ### Example

            If you discover that CloudRift offers DeepSeek-V3 at $0.15 per 1M input token and $0.40 per 1M output token in USD:

            ```text
            write_query with query:
            INSERT INTO pricing_plans
                (company_name, plan_name, input_tokens, output_tokens,
                currency, billing_period, features, limitations, source_query)
            VALUES 
                ('CloudRift', 'DeepSeek-V3', '0.15', '0.40',
                'USD', NULL, NULL, NULL, '<original user query>');
            ```

            Then provide your analysis.

            **Do not output any other text before or after the `write_query` calls; only return the final analysis once all inserts are complete.**
            """

        # Set the model name - use a real Claude model
        model = os.environ.get("ANTHROPIC_BASE_MODEL", "claude-sonnet-4-5-20250929")

        full_response = ""
        process_query = True

        while process_query:
            # Call the Anthropic API
            response = self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=self.all_tools,  # type: ignore
                messages=self.messages,  # type: ignore
            )

            assistant_content = []

            # Process each content block in the response
            for content in response.content:
                if content.type == "text":
                    # Add text to full response
                    full_response += content.text + "\n"
                    # Add to assistant content
                    assistant_content.append(content)

                    # If this is the only content, the model is done
                    if len(response.content) == 1:
                        process_query = False

                elif content.type == "tool_use":
                    # Step 1: Append the tool use request to assistant content
                    assistant_content.append(content)

                    # Add assistant message with tool calls
                    self.messages.append(
                        {"role": "assistant", "content": assistant_content}
                    )

                    # Step 2: Get tool id, args, and name
                    tool_id = content.id
                    tool_name = content.name
                    tool_args = content.input

                    logger.info(f"Tool call: {tool_name}")
                    logger.debug(f"Tool args: {tool_args}")

                    # Step 3: Find the server that has this tool
                    if tool_name in self.tool_to_server:
                        server = self.tool_to_server[tool_name]

                        # Step 4: Execute the tool
                        try:
                            result = await server.execute_tool(tool_name, tool_args)  # type: ignore

                            # Extract text from result
                            if hasattr(result, "content"):
                                result_text = ""
                                for item in result.content:
                                    if hasattr(item, "text"):
                                        result_text += item.text
                                tool_result = result_text
                            else:
                                tool_result = str(result)

                            # Truncate tool result to avoid rate limits (max ~8000 chars)
                            MAX_TOOL_RESULT_LENGTH = 8000
                            if len(tool_result) > MAX_TOOL_RESULT_LENGTH:
                                tool_result = (
                                    tool_result[:MAX_TOOL_RESULT_LENGTH]
                                    + "\n\n[... truncated due to length ...]"
                                )
                                logger.info(
                                    f"Truncated {tool_name} result to {MAX_TOOL_RESULT_LENGTH} chars"
                                )

                        except Exception as e:
                            tool_result = f"Error: {str(e)}"
                            logger.error(f"Tool execution error: {e}")
                    else:
                        tool_result = f"Error: Unknown tool {tool_name}"

                    # Step 5: Append the tool result to messages
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": tool_result,
                                }
                            ],
                        }
                    )

                    # Step 6: Call the model again with new messages
                    response = self.anthropic.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system_prompt,
                        tools=self.all_tools,  # type: ignore
                        messages=self.messages,  # type: ignore
                    )

                    # Step 7: Check if next response is text only
                    if (
                        len(response.content) == 1
                        and response.content[0].type == "text"
                    ):
                        full_response += response.content[0].text + "\n"
                        self.messages.append(
                            {"role": "assistant", "content": response.content}
                        )
                        process_query = False
                    else:
                        # Continue processing - reset assistant_content for next iteration
                        assistant_content = []

        return full_response.strip()

    async def show_stored_data(self):
        """Query and display stored pricing records from SQLite in a clean, readable format."""
        if not self.sqlite_server:
            print("SQLite server not available")
            return

        try:
            # Ensure the pricing_plans table exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS pricing_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT,
                plan_name TEXT,
                input_tokens TEXT,
                output_tokens TEXT,
                currency TEXT,
                billing_period TEXT,
                features TEXT,
                limitations TEXT,
                source_query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            await self.sqlite_server.execute_tool(
                "write_query", {"query": create_table_sql}
            )

            # Execute the read query to get pricing data
            pricing = await self.sqlite_server.execute_tool(
                "read_query",
                {
                    "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC LIMIT 10"
                },
            )

            print("\n" + "=" * 60)
            print("Stored Pricing Data")
            print("=" * 60)

            # The result.content is a list with items containing text
            rows_found = False
            if hasattr(pricing, "content") and pricing.content:
                for item in pricing.content:
                    if hasattr(item, "text"):
                        # Parse the result - may be JSON or Python literal format
                        rows = None
                        try:
                            rows = json.loads(item.text)
                        except json.JSONDecodeError:
                            # SQLite server may return Python repr format (single quotes)
                            try:
                                rows = ast.literal_eval(item.text)
                            except (ValueError, SyntaxError):
                                pass

                        if rows and isinstance(rows, list) and len(rows) > 0:
                            rows_found = True
                            for row in rows:
                                company = row.get("company_name", "Unknown")
                                plan = row.get("plan_name", "Unknown")
                                input_price = row.get("input_tokens", "N/A")
                                output_price = row.get("output_tokens", "N/A")
                                print(
                                    f"• {company}: {plan} - Input Tokens ${input_price}, Output Tokens ${output_price}"
                                )

            if not rows_found:
                print("No pricing data stored yet.")
                print("Use scrape commands to collect pricing data first.")

            print("=" * 60 + "\n")

        except Exception as e:
            print(f"Error querying data: {e}")


async def main():
    """Main entry point for the client application."""
    # Load configuration
    try:
        config = Configuration.load_config("server_config.json")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment")
        print("Please copy .env.example to .env and add your API keys")
        return

    # Initialize Anthropic client
    anthropic_client = Anthropic(api_key=api_key)

    # Initialize servers
    servers: dict[str, Server] = {}
    exit_stack = AsyncExitStack()

    try:
        await exit_stack.__aenter__()

        # Initialize each configured server
        for server_name, server_config in config["mcpServers"].items():
            try:
                server = Server(server_name, server_config)
                await server.initialize(exit_stack)
                servers[server_name] = server
                logger.info(f"Connected to server: {server_name}")
            except Exception as e:
                logger.error(f"Failed to initialize server {server_name}: {e}")
                print(f"Warning: Could not initialize {server_name}: {e}")

        if not servers:
            print("Error: No servers could be initialized")
            return

        # Create chat session
        session = ChatSession(servers, anthropic_client)
        await session.initialize_tools()

        print("\n" + "=" * 80)
        print("PriceScout: The AI-Powered Competitor Analyst")
        print("=" * 80)
        print("Commands:")
        print(" - Type your query in natural language to analyze pricing")
        print(" - 'show data' to see stored pricing records")
        print(" - 'quit' or 'exit' to end the session")
        print("=" * 80 + "\n")

        # Interactive query loop
        while True:
            try:
                user_input = input("Query: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if user_input.lower() == "show data":
                    await session.show_stored_data()
                    continue

                # Process query through the LLM
                print("\nProcessing...\n")
                response = await session.process_query(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nError: {e}\n")

    finally:
        await exit_stack.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
