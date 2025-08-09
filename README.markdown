# ZenDriver MCP Server

A Model Context Protocol (MCP) server for web scraping using ZenDriver's undetectable browser automation. It scrapes websites, extracts unique CSS selectors, and saves results as HTML and CSV files.

## Features
- **Scrape Website**: Downloads full HTML and generates concise CSS selectors (e.g., `div.quote > span.text`).
- **Select Selectors**: Uses OpenAI to pick relevant selectors based on user queries (e.g., "authors and quotes from quotes.toscrape.com").
- **Scrape with Selected Selectors**: Extracts content using chosen selectors with JavaScript fallback for reliability.
- **Execute JavaScript**: Runs custom JavaScript on webpages.
- **Click Elements and Fill Forms**: Interacts with webpage elements.
- **Output**: Saves results in `C:\Users\purna\Downloads\MrScrapper\Claude\Github_MrScrapper\MrScrapper_IndraPurnama\zendriver_mcp`.

## Installation
1. Install dependencies:
   ```bash
   pip install beautifulsoup4 openai
   ```
2. Set OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   On Windows:
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```
3. Install ZenDriver (ensure `zendriver` is available in your environment).

## Usage
Run the server:
```bash
python zendriver_mcp.py
```

### Example Commands
1. **Scrape Website**:
   ```python
   await zendriver_manager.scrape_website(url="https://quotes.toscrape.com")
   ```
   Outputs: `scrape_<timestamp>.html`, `scrape_<timestamp>.csv`, `selectors_<timestamp>.csv`.

2. **Select Selectors**:
   ```python
   await zendriver_manager.select_selectors(query="authors and quotes from quotes.toscrape.com")
   ```
   Outputs: `selected_selectors_<timestamp>.csv`.

3. **Scrape with Selected Selectors**:
   ```python
   await zendriver_manager.scrape_with_selected_selectors(url="https://quotes.toscrape.com")
   ```
   Outputs: `scraped_data_<timestamp>.csv`.

4. **Execute JavaScript**:
   ```python
   await zendriver_manager.execute_javascript(url="https://quotes.toscrape.com", code="return document.querySelectorAll('div.quote').length")
   ```

## Notes
- **Output Directory**: `C:\Users\purna\Downloads\MrScrapper\Claude\Github_MrScrapper\MrScrapper_IndraPurnama\zendriver_mcp`.
- **Error Handling**: Includes fallbacks for selector failures and JavaScript execution.
- **Dependencies**: Requires `zendriver`, `beautifulsoup4`, and `openai`.
- **Troubleshooting**: Check logs for `NoneType` errors or timeouts; increase `wait_timeout` if needed.