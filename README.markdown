# ZenDriver MCP Server

## Overview
The `zendriver_mcp.py` script is a Model Context Protocol (MCP) server that provides web scraping capabilities using ZenDriver. It supports intelligent handling of Single Page Applications (SPAs) and dynamic content, with features like CSS selector extraction, OpenAI-powered selector selection, and JavaScript execution.

## Features
- **Web Scraping**: Extracts HTML content and saves it as HTML and CSV files.
- **SPA Detection**: Intelligently detects and waits for dynamic content to load.
- **Selector Analysis**: Uses OpenAI to select relevant CSS selectors based on user queries.
- **JavaScript Execution**: Executes custom JavaScript on webpages.
- **Form Interaction**: Supports clicking elements and filling out forms.

## Requirements
- Python 3.11+
- Dependencies:
  - `zendriver`: For browser automation
  - `beautifulsoup4`: For HTML parsing
  - `openai`: For AI-powered selector selection
  - `mcp`: For MCP server functionality
- OpenAI API key

## Installation
1. Install Python 3.11 or higher.
2. Install dependencies:
   ```bash
   pip install zendriver beautifulsoup4 openai mcp
   ```
3. Set your OpenAI API key in the script (line ~30):
   ```python
   self.openai_client = AsyncOpenAI(api_key="your-api-key-here")
   ```

## Usage
1. Save the script to `zendriver_mcp.py`.
2. Run the MCP server:
   ```bash
   python zendriver_mcp.py
   ```
3. Interact with the server using an MCP client, invoking tools like:
   - `scrape_website`: Scrape full HTML from a URL.
   - `select_selectors`: Select CSS selectors using OpenAI based on a query.
   - `scrape_with_selected_selectors`: Scrape content using pre-selected selectors.
   - `execute_javascript`, `click_element`, `fill_form`: For advanced interactions.

   Example client command:
   ```json
   {
     "method": "call_tool",
     "params": {
       "name": "scrape_with_selected_selectors",
       "arguments": {
         "url": "https://example.com",
         "query": "jobs and salaries"
       }
     },
     "id": 1
   }
   ```

## Output
- Scraped data is saved to `scraped_data_*.csv` and `scrape_*.html` in the output directory (`C:\Users\purna\Downloads\MrScrapper\Claude\Github_MrScrapper\MrScrapper_IndraPurnama\zendriver_mcp` by default).
- Selector data is saved to `selectors_*.csv` and `selected_selectors_*.csv`.

## Debugging
- Logs are output with `DEBUG` level for detailed troubleshooting.
- Check the output directory for CSV and HTML files.
- Review logs for errors related to invalid selectors or browser issues.

## Notes
- Ensure the OpenAI API key is valid and has sufficient credits.
- The script assumes a Windows environment; adjust paths for other systems.
- Update the `mcp` library if protocol compatibility issues arise:
  ```bash
  pip install --upgrade mcp
  ```