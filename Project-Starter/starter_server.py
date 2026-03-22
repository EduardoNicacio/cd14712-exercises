"""
PriceScout MCP Server - Custom scraper server using Firecrawl.

This module exposes two FastMCP tools:

1. ``scrape_websites``
   Scrapes a list of competitor websites, stores the raw content in
   :data:`SCRAPE_DIR`, and records metadata (including file names,
   title/description, scrape time, etc.) in :data:`METADATA_FILE`.

2. ``extract_scraped_info``
   Retrieves previously scraped data by provider name, URL or domain.
   The function returns a JSON string that contains the stored
   metadata **and** the raw content (if available).

Both tools are automatically registered with FastMCP when this module is
imported; running the file directly starts the MCP server.

The script expects a ``FIRECRAWL_API_KEY`` environment variable.  If it
is missing, the scraper will log an error and return an empty list.

Author: Eduardo Nicacio (eduardo.nicacio @ accenture.com)
"""

import os
import json
import logging
from datetime import datetime
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration & global objects
# --------------------------------------------------------------------------- #

load_dotenv()  # Load environment variables from .env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp = FastMCP("llm_inference")

firecrawl_api_key: str | None = os.getenv("FIRECRAWL_API_KEY")
if not firecrawl_api_key:
    logger.warning("FIRECRAWL_API_KEY not set in environment")
    app = None
else:
    app = FirecrawlApp(api_key=firecrawl_api_key)

# Constants for file paths
METADATA_FILE: str = "scraped_metadata.json"
SCRAPE_DIR: str = "scraped_content"

# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #


@mcp.tool()
def scrape_websites(
    websites: dict[str, str],
    formats: list[str] = ["markdown", "html"],
) -> list[str]:
    """
    Scrape competitor websites and persist the results.

    Parameters
    ----------
    websites:
        Mapping of provider names to URLs.  Example::
            {
                "cloudrift": "https://www.cloudrift.ai/inference",
                "openai":   "https://platform.openai.com/docs"
            }
    formats:
        List of output formats requested from Firecrawl.
        Supported values are ``"markdown"``, ``"html"``, and any other
        format that the Firecrawl API returns.  Defaults to
        ``["markdown", "html"]``.

    Returns
    -------
    list[str]
        The provider names for which a successful scrape was recorded.

    Side effects
    ------------
    * Creates :data:`SCRAPE_DIR` if it does not exist.
    * Writes one file per format for each provider, named
      ``{provider}_{format}.txt`` inside :data:`SCRAPE_DIR`.
    * Updates (or creates) :data:`METADATA_FILE` with a JSON object that
      contains:
        - ``provider``: the key from ``websites``.
        - ``url`` and ``domain`` extracted from the URL.
        - ``scrape_time`` in ISO-8601 format.
        - ``content_files`` mapping each requested format to its file name.
        - ``title`` and ``description`` if Firecrawl supplied them.
      Even failed scrapes are recorded with an empty ``content_files``
      dictionary and an optional ``error`` field.
    * Emits INFO logs for progress, ERROR logs on failures.

    Notes
    -----
    The function will silently return an empty list if the global
    :data:`app` is ``None`` (i.e. no Firecrawl API key).  All other
    exceptions are caught per-provider; a failed provider still gets a
    metadata entry so that future calls to
    :func:`extract_scraped_info` can report the failure.

    Raises
    ------
    None - all errors are logged and handled internally.
    """
    if not app:
        logger.error("Firecrawl API key not configured")
        return []

    # Ensure the scrape directory exists
    os.makedirs(SCRAPE_DIR, exist_ok=True)

    # Load existing metadata or start fresh
    scraped_metadata: dict[str, dict] = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                scraped_metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            scraped_metadata = {}

    successful_scrapes: list[str] = []

    for provider_name, url in websites.items():
        try:
            logger.info(f"Scraping {provider_name}: {url}")

            # Firecrawl API call
            scrape_result = app.scrape(url, formats=formats)

            # Normalise the response to a plain dict
            if hasattr(scrape_result, "model_dump"):
                scrape_result = scrape_result.model_dump()
            elif hasattr(scrape_result, "__dict__"):
                scrape_result = vars(scrape_result)

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            metadata: dict[str, object] = {
                "provider": provider_name,
                "url": url,
                "domain": domain,
                "scrape_time": datetime.now().isoformat(),
                "content_files": {},
                "title": "",
                "description": "",
            }

            if scrape_result.get("success", True):
                for format_type in formats:
                    content = scrape_result.get(format_type, "")
                    if content:
                        filename = f"{provider_name}_{format_type}.txt"
                        filepath = os.path.join(SCRAPE_DIR, filename)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        metadata["content_files"][format_type] = filename # type: ignore

                result_metadata = scrape_result.get("metadata", {})
                metadata["title"] = result_metadata.get("title", "")
                metadata["description"] = result_metadata.get("description", "")

                successful_scrapes.append(provider_name)
                logger.info(f"Successfully scraped {provider_name}")
            else:
                error_msg = scrape_result.get("error", "Unknown error")
                logger.error(f"Failed to scrape {provider_name}: {error_msg}")

            # Record metadata regardless of success
            scraped_metadata[provider_name] = metadata

        except Exception as e:  # pragma: no cover - defensive catch
            logger.error(f"Error scraping {provider_name}: {e}")
            scraped_metadata[provider_name] = {
                "provider": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scrape_time": datetime.now().isoformat(),
                "content_files": {},
                "title": "",
                "description": "",
                "error": str(e),
            }

    # Persist updated metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(scraped_metadata, f, indent=2)

    logger.info(
        f"Scraping complete. Successfully scraped {len(successful_scrapes)} "
        f"out of {len(websites)} websites."
    )

    return successful_scrapes


@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Retrieve previously scraped data by provider name, URL or domain.

    Parameters
    ----------
    identifier:
        A string that identifies the desired scrape.  It can be any of the
        following (case-insensitive for names and URLs):
            * The provider key used in :func:`scrape_websites`.
            * The full URL that was scraped.
            * The domain part of the URL.

    Returns
    -------
    str
        A pretty-printed JSON string containing:
          - All metadata stored during scraping (provider, url,
            domain, scrape_time, title, description, etc.).
          - If available, a ``content`` dictionary mapping each format to
            the raw text read from the corresponding file.
        If no matching entry is found or the metadata file cannot be read,
        an explanatory error message string is returned instead.

    Side effects
    ------------
    * Reads :data:`METADATA_FILE` and any content files in
      :data:`SCRAPE_DIR`.
    * Does **not** modify any state - it is a pure query operation.
    * Emits INFO logs only when the metadata file cannot be read.

    Notes
    -----
    The function performs a simple linear search over all stored entries,
    matching on exact provider name, URL or domain, as well as substring
    matches (case-insensitive).  This is sufficient for small numbers of
    providers but may become slow if the metadata file grows very large.

    Raises
    ------
    None - any I/O errors are caught and reported in the returned string.
    """
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            scraped_metadata = json.load(f)
    except FileNotFoundError:
        return (
            f"There is no saved information related to identifier " f"'{identifier}'."
        )
    except json.JSONDecodeError:
        return (
            f"There is no saved information related to identifier " f"'{identifier}'."
        )

    for provider_name, metadata in scraped_metadata.items():
        if (
            identifier.lower() == provider_name.lower()
            or identifier == metadata.get("url", "")
            or identifier == metadata.get("domain", "")
            or identifier.lower() in metadata.get("url", "").lower()
            or identifier.lower() in metadata.get("domain", "").lower()
        ):
            result = metadata.copy()

            content_files = metadata.get("content_files", {})
            if content_files:
                result["content"] = {}
                for format_type, filename in content_files.items():
                    filepath = os.path.join(SCRAPE_DIR, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            result["content"][format_type] = f.read()
                    except FileNotFoundError:
                        result["content"][format_type] = "File not found"
                    except Exception as e:  # pragma: no cover
                        result["content"][format_type] = f"Error reading file: {e}"

            return json.dumps(result, indent=2)

    return f"There is no saved information related to identifier " f"'{identifier}'."


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run()
