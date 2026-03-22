"""
PriceScout MCP Server - Custom scraper powered by Firecrawl.

This module registers two FastMCP tools that can be invoked from an LLM:

1. **scrape_websites**
   Scrapes a list of competitor sites, stores the raw content in
   :data:`SCRAPE_DIR`, and records metadata (file names, title,
   description, scrape time, etc.) in :data:`METADATA_FILE`.

2. **extract_scraped_info**
   Retrieves previously scraped data by provider name, URL or domain.
   The function returns a JSON string that contains the stored
   metadata *and* the raw content when available.

The server starts automatically when this file is executed directly.
It requires a ``FIRECRAWL_API_KEY`` environment variable; if it is
missing, the scraper logs a warning and returns an empty list.

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

# Load environment variables from .env file
load_dotenv()

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the MCP server with a name
mcp = FastMCP("llm_inference")

# Initialize Firecrawl client with API key from environment
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
if not firecrawl_api_key:
    logger.warning("FIRECRAWL_API_KEY not set in environment")
    app = None
else:
    app = FirecrawlApp(api_key=firecrawl_api_key)

# Constants for file paths
METADATA_FILE = "scraped_metadata.json"
SCRAPE_DIR = "scraped_content"


@mcp.tool()
def scrape_websites(
    websites: dict[str, str], formats: list[str] = ["markdown", "html"]
) -> list[str]:
    """
    Scrape competitor websites and persist the results.

    Parameters
    ----------
    websites : dict[str, str]
        Mapping of provider names to their URLs.
        Example: ``{'cloudrift': 'https://www.cloudrift.ai/inference'}``
    formats : list[str], optional
        Content formats requested from Firecrawl. Defaults to
        ``['markdown', 'html']``.

    Returns
    -------
    list[str]
        Names of providers that were scraped successfully.

    Notes
    -----
    * The function writes raw content files into :data:`SCRAPE_DIR`.
      Filenames follow the pattern ``{provider}_{format}.txt``.
    * Metadata for each provider is stored in :data:`METADATA_FILE`,
      including file names, title, description, and scrape timestamp.
    * If the Firecrawl API key is missing or a request fails,
      an error is logged and the provider is omitted from the
      returned list.  A metadata entry with an ``error`` field is still
      created to record the failure.

    Raises
    ------
    None - all exceptions are caught internally; failures are logged.
    """
    if not app:
        logger.error("Firecrawl API key not configured")
        return []

    # Ensure the scrape directory exists
    if not os.path.exists(SCRAPE_DIR):
        os.makedirs(SCRAPE_DIR)

    # Load existing metadata or initialize empty dict
    scraped_metadata = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                scraped_metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            scraped_metadata = {}

    # Track successful scrapes
    successful_scrapes = []

    # Loop through each website to scrape
    for provider_name, url in websites.items():
        try:
            logger.info(f"Scraping {provider_name}: {url}")

            # Call Firecrawl API to scrape the URL
            scrape_result = app.scrape(url, formats=formats)

            # Handle both dict and object responses
            if hasattr(scrape_result, "model_dump"):
                scrape_result = scrape_result.model_dump()
            elif hasattr(scrape_result, "__dict__"):
                scrape_result = vars(scrape_result)

            # Extract domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Create metadata for this provider
            metadata = {
                "provider": provider_name,
                "url": url,
                "domain": domain,
                "scrape_time": datetime.now().isoformat(),
                "content_files": {},
                "title": "",
                "description": "",
            }

            # Check if scrape was successful
            if scrape_result.get(
                "success", True
            ):  # Default True for backwards compatibility
                # Save content for each format
                for format_type in formats:
                    content = scrape_result.get(format_type, "")
                    if content:
                        # Create filename: {provider}_{format}.txt
                        filename = f"{provider_name}_{format_type}.txt"
                        filepath = os.path.join(SCRAPE_DIR, filename)

                        # Write content to file
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)

                        # Track the file in metadata
                        metadata["content_files"][format_type] = filename

                # Extract title and description from metadata if available
                result_metadata = scrape_result.get("metadata", {})
                metadata["title"] = result_metadata.get("title", "")
                metadata["description"] = result_metadata.get("description", "")

                # Add to successful scrapes
                successful_scrapes.append(provider_name)
                logger.info(f"Successfully scraped {provider_name}")

            else:
                # Log the error from scrape result
                error_msg = scrape_result.get("error", "Unknown error")
                logger.error(f"Failed to scrape {provider_name}: {error_msg}")

            # Always add metadata entry (even if failed, for tracking)
            scraped_metadata[provider_name] = metadata

        except Exception as e:
            logger.error(f"Error scraping {provider_name}: {e}")
            # Still create a metadata entry to track the attempt
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

    # Write updated metadata to file
    with open(METADATA_FILE, "w") as f:
        json.dump(scraped_metadata, f, indent=2)

    logger.info(
        f"Scraping complete. Successfully scraped {len(successful_scrapes)} out of {len(websites)} websites."
    )

    return successful_scrapes


@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Retrieve stored scraping results for a given provider, URL or domain.

    Parameters
    ----------
    identifier : str
        The provider name, full URL, or domain to search for.  Matching is
        case-insensitive and supports partial matches on URLs/domains.

    Returns
    -------
    str
        A JSON string containing the metadata and any available raw
        content files.  If no matching entry exists, a user-friendly error
        message is returned instead of raising an exception.

    Notes
    -----
    * The function reads :data:`METADATA_FILE` to locate the requested
      provider.  It then loads any associated content files from
      :data:`SCRAPE_DIR`.
    * If the metadata file cannot be read or contains invalid JSON,
      a clear message is returned indicating that no data was found.
    """
    try:
        # Load metadata file
        with open(METADATA_FILE, "r") as f:
            scraped_metadata = json.load(f)
    except FileNotFoundError:
        return f"There's no saved information related to identifier '{identifier}'."
    except json.JSONDecodeError:
        return f"There's no saved information related to identifier '{identifier}'."

    # Search for a matching entry
    for provider_name, metadata in scraped_metadata.items():
        # Check if identifier matches provider name, URL, or domain
        if (
            identifier.lower() == provider_name.lower()
            or identifier == metadata.get("url", "")
            or identifier == metadata.get("domain", "")
            or identifier.lower() in metadata.get("url", "").lower()
            or identifier.lower() in metadata.get("domain", "").lower()
        ):

            # Found a match - make a copy to add content
            result = metadata.copy()

            # Load content from files if available
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
                    except Exception as e:
                        result["content"][format_type] = f"Error reading file: {e}"

            # Return formatted JSON string
            return json.dumps(result, indent=2)

    # No match found
    return f"There's no saved information related to identifier '{identifier}'."


# Run the server when this file is executed directly
if __name__ == "__main__":
    mcp.run()
