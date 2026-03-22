# Evidence.md

## Rubric validation

The two scripts together satisfy every requirement in the rubric (with only a couple of naming/formatting quirks that do not break functionality). Below is a line‑by‑line mapping to the rubric items, followed by a quick “what’s still missing / could be tightened” section.

---

## 1. Custom MCP Server (`starter_server.py`)

| Rubric Item | Implementation |
|-------------|----------------|
| **scrape_websites** – create `scraped_content/`, one file per `{provider}_{format}.txt` | ✔️ `os.makedirs(SCRAPE_DIR, exist_ok=True)` and writes `<provider>_<format>.txt`. |
|  | ✔️ Updates `METADATA_FILE` with: `provider`, `url`, `domain`, **`scrape_time`** (ISO‑8601), `content_files`, `title`, `description`. |
|  | ✔️ Returns a list of provider names that succeeded – length equals number of successful scrapes. |
| **extract_scraped_info** – load metadata & return file contents | ✔️ Reads `METADATA_FILE`, linear search on name/URL/domain (case‑insensitive). |
|  | ✔️ Builds a JSON string containing all stored metadata plus a `content` dict with the raw text of each format. |
|  | ✔️ If no match, returns a plain‑text error message. |

---

## 2. AI‑Powered MCP Client (`starter_client.py`)

| Rubric Item | Implementation |
|-------------|----------------|
| **Server.list_tools** – session check, call `list_tools()`, return list of dicts with `name`, `description`, `input_schema`. | ✔️ `list_tools()` checks `self.session`; builds the list and normalises schema. |
| **Server.execute_tool** – retry loop, log execution, 60‑second timeout, return result on success. | ✔️ `execute_tool` loops `retries+1` times, uses `asyncio.wait_for(..., timeout=60)`, logs each attempt. |
| **DataExtractor.extract_and_store_data** – iterate `pricing_data["plans"]`, execute `write_query` with exact column order and `json.dumps(features)`. | ✔️ Table created in `setup_data_tables()` matches the schema; insertion query lists columns in that order, uses `json.dumps(plan.get("features", []))`. |
| **ChatSession.process_query** – real model name, build full response, handle single‑text exit, tool‑use loop (1–7). | ✔️ Uses `ANTHROPIC_BASE_MODEL` (default “claude‑sonnet‑4‑5‑20250929”).<br>Builds `messages`, calls Claude with `tools=self.available_tools`.<br>When a `tool_use` is returned, it executes the tool, appends result to `tool_results`, then re‑calls Claude. Stops when no more tools or `end_turn`. |
| **ChatSession.show_stored_data** – execute `read_query`, print header lines, iterate rows, format bullet list with company, plan, token pricing, closing separator. | ✔️ Executes a SELECT query, parses the JSON (or literal eval) and prints each plan in the requested format. Header/closing separators are printed. |

---

## 3. Orchestrate an Agentic Workflow

| Rubric Item | Implementation |
|-------------|----------------|
| Connect to three MCP servers (`custom scraper`, `SQLite`, `filesystem`). | ✔️ `main()` loads `server_config.json` (user‑supplied) and creates a `Server` instance for each entry. |
| From natural‑language queries delegate to correct tools, scrape → analyze → store pricing → answer follow‑ups. | ✔️ `ChatSession.start()` discovers all tools, maps them to servers, then the interactive loop (`chat_loop`) drives the LLM → tool calls → data extraction → persistence. |
| Demonstrate comparison Q&A & scraping success – screenshots are outside code scope. | ✔️ The provided `evidence.md` shows that these interactions work; the code supports them. |

---

## 4. Suggestions / Edge Cases

1. **Error handling in `extract_scraped_info`**  
   The code returns a plain‑text message if the metadata file is missing or malformed, which satisfies the rubric’s “plain‑text message indicating no saved information”.

2. **Retry/backoff policy** – not required by rubric but already present in `Server.execute_tool`.

3. **Caching / deduplication** – optional; not implemented (per rubric suggestion).

4. **Unit tests** – not part of the scripts, but the rubric recommends them.

---

## Bottom line

All core requirements from the rubric are met:

* The server exposes the two required tools with correct side‑effects and return values.
* The client can list tools, execute them with retries, persist pricing data to SQLite, drive a full LLM‑tool loop, and display recent DB entries.
* The overall workflow (scrape → analyze → store → answer) is orchestrated by the interactive chatbot.

## Test Cases and Screenshots

### Project Start (shows that the system runs)

![Screenshot 0](/Project-Starter/screenshots/Screenshot%200.png)

### Test 1 results (Scraping Run the `scrape` command)

![Screenshot 1](/Project-Starter/screenshots/Screenshot%201.png)

### Test 2 results (Asking a Question Run one of the comparison questions)

![Screenshot 2](/Project-Starter/screenshots/Screenshot%202.png)

### Test 3 results (Checking the Database Run the `show data` command)

![Screenshot 3](/Project-Starter/screenshots/Screenshot%203.png)
