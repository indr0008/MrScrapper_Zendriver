#!/usr/bin/env python3
"""
Enhanced ZenDriver MCP Server with Smart SPA/Dynamic Content Handling

A Model Context Protocol server that provides web scraping capabilities
with intelligent detection and handling of Single Page Applications.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import csv
import os
from collections import defaultdict
from bs4 import BeautifulSoup
import time

try:
    import zendriver as zd
except ImportError:
    raise ImportError("zendriver is required. Install with: pip install zendriver")
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("beautifulsoup4 is required. Install with: pip install beautifulsoup4")
try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai is required. Install with: pip install openai")

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed logs
logger = logging.getLogger("zendriver-mcp")

class ZenDriverManager:
    def __init__(self):
        self.browser = None
        self.pages = {}  # Store active pages by ID
        self.output_dir = Path(r"C:\Users\purna\Downloads\MrScrapper\Claude\Github_MrScrapper\MrScrapper_IndraPurnama\zendriver_mcp")
        self.output_dir.mkdir(exist_ok=True)
        self.openai_client = AsyncOpenAI(api_key="sk-xxxx")

    def is_valid_css_selector(self, selector: str) -> bool:
        """Check if a string is a valid CSS selector."""
        if not selector or selector.strip() in ['```', '``', '']:
            return False
        # Basic validation: check for common invalid patterns
        invalid_patterns = ['```', '``', '\n', '\r', '\t']
        return not any(pattern in selector for pattern in invalid_patterns)

    async def _ensure_browser(self):
        """Ensure browser is started"""
        if self.browser is None:
            logger.info("Starting ZenDriver browser...")
            self.browser = await zd.start()
            
    async def _get_page(self, url: str):
        """Get or create a page for the given URL"""
        await self._ensure_browser()
        
        # Use URL as page key for simplicity
        page_key = url
        
        if page_key not in self.pages:
            logger.info(f"Creating new page for {url}")
            page = await self.browser.get(url)
            self.pages[page_key] = page
        else:
            page = self.pages[page_key]
            # Navigate to URL if it's different
            if page.url != url:
                await page.get(url)
                
        return self.pages[page_key]

    async def _detect_spa_and_wait_for_content(self, page, wait_timeout: float = 10) -> Dict[str, Any]:
        """
        Intelligently detect if page is SPA and wait for content to load
        Returns info about the page and content loading status
        """
        spa_detection_js = """
        (function() {
            const info = {
                isSPA: false,
                framework: 'unknown',
                initialContentLength: document.body.innerText.length,
                hasReactRoot: false,
                hasVueApp: false,
                hasAngularApp: false,
                hasLoadingIndicators: false,
                dynamicElements: 0
            };
            
            // Check for React
            if (document.querySelector('#__next, [data-reactroot], [data-react-helmet]') || 
                window.React || window._React || document.querySelector('script[src*="react"]')) {
                info.isSPA = true;
                info.framework = 'React';
                info.hasReactRoot = true;
            }
            
            // Check for Vue
            if (document.querySelector('#app, [data-v-]') || window.Vue || 
                document.querySelector('script[src*="vue"]')) {
                info.isSPA = true;
                info.framework = 'Vue';
                info.hasVueApp = true;
            }
            
            // Check for Angular
            if (document.querySelector('[ng-app], [ng-version]') || window.angular || 
                document.querySelector('script[src*="angular"]')) {
                info.isSPA = true;
                info.framework = 'Angular';
                info.hasAngularApp = true;
            }
            
            // Check for loading indicators
            const loadingSelectors = [
                '[class*="loading"]', '[class*="spinner"]', '[class*="skeleton"]',
                '[class*="placeholder"]', '[id*="loading"]', '[data-testid*="loading"]'
            ];
            
            for (const selector of loadingSelectors) {
                if (document.querySelector(selector)) {
                    info.hasLoadingIndicators = true;
                    break;
                }
            }
            
            // Count elements with dynamic-looking classes
            const dynamicPatterns = ['card', 'item', 'job', 'product', 'post', 'article'];
            for (const pattern of dynamicPatterns) {
                const elements = document.querySelectorAll(`[class*="${pattern}"]`);
                info.dynamicElements += elements.length;
            }
            
            return info;
        })();
        """
        
        try:
            # Get initial page info
            page_info = await page.evaluate(spa_detection_js)
            logger.info(f"Page analysis: {page_info}")
            
            # If it's an SPA or has minimal content, wait for content to load
            if page_info.get('isSPA', False) or page_info.get('initialContentLength', 0) < 1000:
                logger.info(f"Detected {page_info.get('framework', 'SPA')} application, waiting for content...")
                
                # Wait for network idle first
                try:
                    await page.wait_for_load_state('networkidle', timeout=wait_timeout * 1000)
                except Exception as e:
                    logger.warning(f"Network idle timeout: {e}")
                
                # Smart content waiting with multiple strategies
                await self._smart_content_wait(page, wait_timeout)
                
                # Re-evaluate content after waiting
                updated_info = await page.evaluate(spa_detection_js)
                page_info.update(updated_info)
                
            return page_info
            
        except Exception as e:
            logger.error(f"Error in SPA detection: {e}")
            return {"error": str(e), "isSPA": False}

    async def _smart_content_wait(self, page, timeout: float):
        """
        Smart waiting strategy for dynamic content
        """
        content_wait_js = """
        (function() {
            return new Promise((resolve) => {
                let attempts = 0;
                const maxAttempts = 30; // 3 seconds max (100ms intervals)
                
                const checkContent = () => {
                    attempts++;
                    
                    // Check for meaningful content
                    const bodyText = document.body.innerText.length;
                    const hasJobCards = document.querySelectorAll('[class*="job"], [class*="card"], [class*="item"]').length;
                    const hasListItems = document.querySelectorAll('li, article, [role="listitem"]').length;
                    const noLoadingIndicators = !document.querySelector('[class*="loading"], [class*="spinner"]');
                    
                    // Content is ready if we have substantial text and elements
                    if (bodyText > 2000 && (hasJobCards > 5 || hasListItems > 10) && noLoadingIndicators) {
                        resolve({
                            success: true,
                            contentLength: bodyText,
                            dynamicElements: hasJobCards + hasListItems,
                            attempts: attempts
                        });
                        return;
                    }
                    
                    // Keep trying until timeout
                    if (attempts < maxAttempts) {
                        setTimeout(checkContent, 100);
                    } else {
                        resolve({
                            success: false,
                            contentLength: bodyText,
                            dynamicElements: hasJobCards + hasListItems,
                            attempts: attempts
                        });
                    }
                };
                
                checkContent();
            });
        })();
        """
        
        try:
            result = await page.evaluate(content_wait_js, timeout=timeout * 1000)
            logger.info(f"Content wait result: {result}")
            return result
        except Exception as e:
            logger.warning(f"Smart content wait failed: {e}")
            return {"success": False, "error": str(e)}

    async def _extract_enhanced_selectors(self, page) -> Dict[str, List[str]]:
        """
        Use JavaScript to extract better selectors from the fully loaded page
        """
        selector_extraction_js = """
        (function() {
            const selectorMap = {};
            
            // Enhanced selector discovery
            const elements = document.querySelectorAll('*');
            const textElements = [];
            
            elements.forEach(elem => {
                const text = elem.innerText?.trim();
                if (text && text.length > 5 && text.length < 200) {
                    // Skip elements that are just containers
                    const childTextLength = Array.from(elem.children)
                        .reduce((sum, child) => sum + (child.innerText?.length || 0), 0);
                    
                    if (text.length > childTextLength * 1.5) {
                        textElements.push({
                            element: elem,
                            text: text,
                            tag: elem.tagName.toLowerCase(),
                            classes: Array.from(elem.classList),
                            id: elem.id
                        });
                    }
                }
            });
            
            // Group by content patterns
            textElements.forEach(item => {
                let selector = item.tag;
                
                // Build specific selector
                if (item.id) {
                    selector = `${item.tag}#${item.id}`;
                } else if (item.classes.length > 0) {
                    // Use most specific class
                    const specificClass = item.classes.find(cls => 
                        cls.includes('title') || cls.includes('price') || cls.includes('name') ||
                        cls.includes('job') || cls.includes('card') || cls.includes('item')
                    ) || item.classes[0];
                    selector = `${item.tag}.${specificClass}`;
                }
                
                if (!selectorMap[selector]) {
                    selectorMap[selector] = [];
                }
                
                if (selectorMap[selector].length < 3) {
                    selectorMap[selector].push(item.text);
                }
            });
            
            return selectorMap;
        })();
        """
        
        try:
            selectors = await page.evaluate(selector_extraction_js)
            logger.info(f"Extracted {len(selectors)} enhanced selectors")
            return selectors
        except Exception as e:
            logger.error(f"Error extracting enhanced selectors: {e}")
            return {}

    async def scrape_website(self, url: str, wait_for: str = None, 
                             wait_timeout: float = 10, execute_js: str = None) -> Dict[str, str]:
        """
        Enhanced scrape with intelligent SPA handling and dynamic content detection
        """
        page = await self._get_page(url)

        # Step 1: Detect SPA and wait for content intelligently
        page_info = await self._detect_spa_and_wait_for_content(page, wait_timeout)
        
        # Step 2: Handle custom wait conditions
        if wait_for:
            try:
                if wait_for.startswith('#') or wait_for.startswith('.') or ' ' in wait_for:
                    # CSS selector
                    await page.select(wait_for, timeout=wait_timeout)
                else:
                    # Text content
                    await page.find(wait_for, timeout=wait_timeout)
                logger.info(f"Successfully waited for: {wait_for}")
            except Exception as e:
                logger.warning(f"Wait condition not met: {e}")

        # Step 3: Execute custom JavaScript if provided
        if execute_js:
            try:
                await page.evaluate(execute_js)
                logger.info("Custom JavaScript executed successfully")
            except Exception as e:
                logger.warning(f"JavaScript execution failed: {e}")

        # Step 4: Final content verification and extraction
        final_content_check = await page.evaluate("""
            ({
                contentLength: document.body.innerText.length,
                elementCount: document.querySelectorAll('*').length,
                hasContent: document.body.innerText.length > 1000,
                url: window.location.href,
                title: document.title
            })
        """)
        
        logger.info(f"Final content check: {final_content_check}")

        # Extract full HTML content
        content = []
        full_html = ""
        try:
            full_html = await page.get_content()
            content.append(f"Successfully scraped {url}")
            content.append(f"Page Info: {json.dumps(page_info, indent=2)}")
            content.append(f"Content Stats: {json.dumps(final_content_check, indent=2)}")
        except Exception as e:
            logger.warning(f"Error getting full page content: {e}")
            full_html = f"Error getting page content: {e}"
            content.append(full_html)

        # Generate filenames based on timestamp
        timestamp = int(time.time())
        html_filename = f"scrape_{timestamp}.html"
        csv_filename = f"scrape_{timestamp}.csv"
        selectors_csv_filename = f"selectors_{timestamp}.csv"
        html_filepath = self.output_dir / html_filename
        csv_filepath = self.output_dir / csv_filename
        selectors_csv_filepath = self.output_dir / selectors_csv_filename

        # Save HTML content
        try:
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(full_html)
            logger.info(f"Saved HTML content to {html_filepath}")
        except Exception as e:
            logger.error(f"Error saving HTML file: {e}")
            content.append(f"Error saving HTML file: {e}")

        # Save content as CSV
        try:
            with open(csv_filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['URL', 'Content', 'PageInfo'])
                writer.writerow([url, full_html, json.dumps(page_info)])
            logger.info(f"Saved CSV content to {csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving CSV file: {e}")
            content.append(f"Error saving CSV file: {e}")

        # Enhanced selector analysis
        selector_examples = defaultdict(list)
        try:
            # First try JavaScript-based enhanced extraction
            enhanced_selectors = await self._extract_enhanced_selectors(page)
            
            # Add enhanced selectors to our collection
            for selector, examples in enhanced_selectors.items():
                if len(selector_examples[selector]) < 2:
                    selector_examples[selector].extend(examples[:2])
            
            # Fallback to BeautifulSoup analysis
            soup = BeautifulSoup(full_html, 'html.parser')
            elements = soup.find_all(True)
            
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and 5 <= len(text) <= 200:
                    # Construct CSS selector
                    selector = elem.name
                    if elem.get('id'):
                        selector += f"#{elem.get('id')}"
                    if elem.get('class'):
                        classes = '.'.join(elem.get('class')).replace(' ', '.')
                        if classes:
                            selector += f".{classes}"
                    
                    if selector and len(selector_examples[selector]) < 2:
                        selector_examples[selector].append(text)
                        
        except Exception as e:
            logger.warning(f"Error analyzing HTML for selectors: {e}")
            content.append(f"Error analyzing HTML for selectors: {str(e)}")

        # Save enhanced selectors CSV
        if selector_examples:
            try:
                with open(selectors_csv_filepath, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Selector', 'Example1', 'Example2', 'Source'])
                    for sel, examples in selector_examples.items():
                        example1 = examples[0] if examples else ""
                        example2 = examples[1] if len(examples) > 1 else ""
                        source = "enhanced" if sel in enhanced_selectors else "beautifulsoup"
                        writer.writerow([sel, example1, example2, source])
                logger.info(f"Saved enhanced selectors CSV to {selectors_csv_filepath}")
            except Exception as e:
                logger.error(f"Error saving selectors CSV file: {e}")
                content.append(f"Error saving selectors CSV file: {e}")

        result = "\n".join(content) if content else "No content extracted"
        return {
            "result": result,
            "selectors_csv": str(selectors_csv_filepath),
            "page_info": page_info,
            "content_stats": final_content_check
        }

    async def select_selectors(self, query: str, selectors_csv: str = None) -> str:
        """Use OpenAI API to select appropriate CSS selectors based on user query"""
        # Find the latest selectors CSV if not provided
        if not selectors_csv:
            selectors_files = list(self.output_dir.glob("selectors_*.csv"))
            if not selectors_files:
                return "Error: No selectors CSV files found in output directory"
            selectors_csv = str(max(selectors_files, key=os.path.getmtime))
            logger.info(f"Using latest selectors CSV: {selectors_csv}")
        else:
            if not os.path.exists(selectors_csv):
                return f"Error: Selectors CSV file not found: {selectors_csv}"
        
        # Read selectors and examples from CSV
        selectors_data = []
        try:
            with open(selectors_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    selectors_data.append({
                        "selector": row["Selector"],
                        "example1": row["Example1"],
                        "example2": row["Example2"],
                        "source": row.get("Source", "unknown")
                    })
        except Exception as e:
            logger.error(f"Error reading selectors CSV: {e}")
            return f"Error reading selectors CSV: {e}"
        
        if not selectors_data:
            return "Error: No selectors found in CSV file"
        
        # Enhanced prompt for OpenAI
        prompt = f"""
You are a web scraping expert. Given a user query and CSS selectors with examples from a webpage, identify the most appropriate selectors.

User Query: {query}

Available Selectors:
{json.dumps(selectors_data, indent=2)}

Instructions:
1. Look for selectors whose examples match the user's query
2. Prefer "enhanced" source selectors as they're more accurate
3. Return only the most relevant selectors, one per line
4. Do not include markdown code block delimiters (```) or any other formatting
5. Ensure each selector is a valid CSS selector

Output only the selectors, one per line.
"""
        
        # Call OpenAI API
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a web scraping assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            # Sanitize the response to remove markdown delimiters and invalid selectors
            raw_selectors = response.choices[0].message.content.strip()
            logger.info(f"Raw OpenAI response: {raw_selectors}")
            selected_selectors = [
                s.strip() for s in raw_selectors.split("\n")
                if s.strip() and self.is_valid_css_selector(s.strip())
            ]
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error calling OpenAI API: {e}"
        
        if not selected_selectors:
            return "No valid selectors matched the query"
        
        # Save selected selectors to a new CSV
        timestamp = int(time.time())
        selected_csv_filename = f"selected_selectors_{timestamp}.csv"
        selected_csv_filepath = self.output_dir / selected_csv_filename
        try:
            with open(selected_csv_filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Selector', 'Example1', 'Example2', 'Query', 'Source'])
                for selector in selected_selectors:
                    # Find the original selector data
                    for data in selectors_data:
                        if data["selector"] == selector or selector.startswith(data["selector"]):
                            writer.writerow([selector, data["example1"], data["example2"], query, data.get("source", "unknown")])
                            break
                    else:
                        writer.writerow([selector, "", "", query, "unknown"])
            logger.info(f"Saved selected selectors CSV to {selected_csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving selected selectors CSV: {e}")
            return f"Error saving selected selectors CSV: {e}"
        
        return f"Selected selectors saved to {selected_csv_filepath}\n" + "\n".join(selected_selectors)

    async def scrape_with_selected_selectors(self, url: str, selected_csv: str = None, wait_for: str = None, 
                                            wait_timeout: float = 30) -> str:
        """Scrape webpage content using selectors from selected_selectors CSV and save results"""
        # Find the latest selected selectors CSV if not provided
        if not selected_csv:
            selected_files = list(self.output_dir.glob("selected_selectors_*.csv"))
            if not selected_files:
                return "Error: No selected selectors CSV files found in output directory"
            selected_csv = str(max(selected_files, key=os.path.getmtime))
            logger.info(f"Using latest selected selectors CSV: {selected_csv}")
        else:
            if not os.path.exists(selected_csv):
                return f"Error: Selected selectors CSV file not found: {selected_csv}"
        
        # Read selectors and query from CSV
        selectors_data = []
        try:
            with open(selected_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    selectors_data.append({
                        "selector": row["Selector"],
                        "query": row["Query"],
                        "source": row.get("Source", "unknown")
                    })
        except Exception as e:
            logger.error(f"Error reading selected selectors CSV: {e}")
            return f"Error reading selected selectors CSV: {e}"
        
        if not selectors_data:
            return "Error: No selectors found in selected selectors CSV"
        
        # Navigate to the webpage and ensure content is loaded
        page = await self._get_page(url)
        
        # Use our enhanced SPA detection and waiting
        page_info = await self._detect_spa_and_wait_for_content(page, wait_timeout)
        
        # Wait for specific element if specified
        if wait_for:
            try:
                if wait_for.startswith('#') or wait_for.startswith('.') or ' ' in wait_for:
                    await page.select(wait_for, timeout=wait_timeout)
                else:
                    await page.find(wait_for, timeout=wait_timeout)
            except Exception as e:
                logger.warning(f"Wait condition not met: {e}")
        
        # Scrape content for each selector with enhanced error handling
        scraped_data = []
        for data in selectors_data:
            selector = data["selector"]
            query = data["query"]
            source = data.get("source", "unknown")
            
            # Validate the selector before processing
            if not self.is_valid_css_selector(selector):
                logger.error(f"Invalid CSS selector: {selector}")
                scraped_data.append({
                    "selector": selector,
                    "content": f"Error: Invalid CSS selector '{selector}'",
                    "query": query,
                    "source": source,
                    "method": "error"
                })
                continue
            
            try:
                # Try JavaScript-based extraction first (more reliable for SPAs)
                # Escape single quotes in the selector for JavaScript
                escaped_selector = selector.replace("'", "\\'")
                js_code = """
                (function() {
                    const elements = document.querySelectorAll('""" + escaped_selector + """');
                    const results = [];
                    elements.forEach((elem, index) => {
                        if (index < 50) { // Limit to 50 elements to avoid overwhelming
                            const text = elem.innerText?.trim() || elem.textContent?.trim() || '';
                            if (text && text.length > 2) {
                                results.push(text);
                            }
                        }
                    });
                    return results;
                })();
                """
                
                js_results = await page.evaluate(js_code)
                
                if js_results and len(js_results) > 0:
                    for text in js_results:
                        scraped_data.append({
                            "selector": selector, 
                            "content": text, 
                            "query": query,
                            "source": source,
                            "method": "javascript"
                        })
                    logger.info(f"Successfully extracted {len(js_results)} items with selector: {selector}")
                else:
                    # Fallback to ZenDriver selector method
                    try:
                        elements = await page.select_all(selector)
                        if elements:
                            for elem in elements[:20]:  # Limit to 20 elements
                                try:
                                    text = await elem.get_text()
                                    if text and text.strip():
                                        scraped_data.append({
                                            "selector": selector,
                                            "content": text.strip(),
                                            "query": query,
                                            "source": source,
                                            "method": "zendriver"
                                        })
                                except Exception as e:
                                    logger.warning(f"Error getting text from element: {e}")
                        else:
                            scraped_data.append({
                                "selector": selector,
                                "content": "No elements found",
                                "query": query,
                                "source": source,
                                "method": "zendriver"
                            })
                    except Exception as e:
                        scraped_data.append({
                            "selector": selector,
                            "content": f"Error: {str(e)}",
                            "query": query,
                            "source": source,
                            "method": "error"
                        })
                        
            except Exception as e:
                logger.error(f"Error scraping with selector {selector}: {e}")
                scraped_data.append({
                    "selector": selector,
                    "content": f"Error: {str(e)}",
                    "query": query,
                    "source": source,
                    "method": "error"
                })
        
        if not scraped_data:
            return "No content scraped for the provided selectors"
        
        # Save scraped data to a new CSV with enhanced metadata
        timestamp = int(time.time())
        scraped_csv_filename = f"scraped_data_{timestamp}.csv"
        scraped_csv_filepath = self.output_dir / scraped_csv_filename
        try:
            with open(scraped_csv_filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Selector', 'Content', 'Query', 'Source', 'Method'])
                for item in scraped_data:
                    writer.writerow([
                        item["selector"], 
                        item["content"], 
                        item["query"],
                        item["source"],
                        item["method"]
                    ])
            logger.info(f"Saved scraped data CSV to {scraped_csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving scraped data CSV: {e}")
            return f"Error saving scraped data CSV: {e}"
        
        # Create summary
        unique_selectors = len(set(item["selector"] for item in scraped_data))
        total_items = len(scraped_data)
        successful_items = len([item for item in scraped_data if not item["content"].startswith("Error")])
        
        result = f"""Scraped data saved to {scraped_csv_filepath}
Summary:
- Total selectors processed: {unique_selectors}
- Total items extracted: {total_items}
- Successful extractions: {successful_items}
- Page type: {page_info.get('framework', 'Unknown')}

Sample data:
""" + "\n".join([f"{item['selector']}: {item['content'][:100]}..." for item in scraped_data[:10]])
        
        return result

    async def execute_javascript(self, url: str, code: str, wait_for_result: bool = True) -> str:
        """Execute JavaScript on a webpage"""
        page = await self._get_page(url)
        
        try:
            # Wrap code in a function to avoid illegal return statements
            wrapped_code = f"(function() {{ {code} }})()"
            if wait_for_result:
                result = await page.evaluate(wrapped_code)
                return f"JavaScript result: {json.dumps(result)}"
            else:
                await page.evaluate(wrapped_code)
                return "JavaScript executed successfully"
        except Exception as e:
            return f"Error executing JavaScript: {e}"

    async def click_element(self, url: str, selector: str = None, text: str = None, 
                            wait_after_click: float = 1) -> str:
        """Click on an element"""
        page = await self._get_page(url)
        
        try:
            if selector:
                if not self.is_valid_css_selector(selector):
                    return f"Error: Invalid CSS selector '{selector}'"
                element = await page.select(selector)
            elif text:
                element = await page.find(text)
            else:
                return "Error: Must provide either 'selector' or 'text' parameter"
            
            await element.click()
            await asyncio.sleep(wait_after_click)
            
            return "Element clicked successfully"
            
        except Exception as e:
            return f"Error clicking element: {e}"

    async def fill_form(self, url: str, fields: Dict[str, str], submit: bool = False, 
                        submit_selector: str = None) -> str:
        """Fill out form fields"""
        page = await self._get_page(url)
        
        try:
            results = []
            
            for field_selector, value in fields.items():
                try:
                    if not self.is_valid_css_selector(field_selector):
                        results.append(f"Error: Invalid CSS selector '{field_selector}'")
                        continue
                    element = await page.select(field_selector)
                    await element.send_keys(value)
                    results.append(f"Filled field '{field_selector}' with '{value}'")
                except Exception as e:
                    results.append(f"Error filling field '{field_selector}': {e}")
            
            if submit:
                try:
                    if submit_selector:
                        if not self.is_valid_css_selector(submit_selector):
                            results.append(f"Error: Invalid submit selector '{submit_selector}'")
                        else:
                            submit_btn = await page.select(submit_selector)
                            await submit_btn.click()
                            results.append("Form submitted successfully")
                    else:
                        # Try to find submit button
                        submit_btn = await page.select('input[type="submit"], button[type="submit"], button:contains("Submit")')
                        await submit_btn.click()
                        results.append("Form submitted successfully")
                except Exception as e:
                    results.append(f"Error submitting form: {e}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error filling form: {e}"

    async def cleanup(self):
        """Clean up resources"""
        if self.browser:
            try:
                await self.browser.stop()
            except Exception as e:
                logger.error(f"Error stopping browser: {e}")
            self.browser = None
        self.pages.clear()

# Create the MCP server
server = Server("zendriver-scraper")
zendriver_manager = ZenDriverManager()

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools"""
    return [
        types.Tool(
            name="scrape_website",
            description="Scrape full HTML from a website with intelligent SPA detection and dynamic content handling",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "wait_for": {"type": "string", "description": "CSS selector or text to wait for before scraping (optional)"},
                    "wait_timeout": {"type": "number", "description": "Timeout in seconds to wait for elements (default: 10)", "default": 10},
                    "execute_js": {"type": "string", "description": "JavaScript code to execute before scraping (optional)"}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="select_selectors",
            description="Use OpenAI API to select appropriate CSS selectors based on a user query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query describing desired content (e.g., 'authors and quotes from quotes.toscrape.com')"},
                    "selectors_csv": {"type": "string", "description": "Path to selectors CSV file (optional, uses latest if not provided)"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="scrape_with_selected_selectors",
            description="Scrape webpage content using selectors from a selected_selectors CSV file with enhanced SPA support",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "selected_csv": {"type": "string", "description": "Path to selected selectors CSV file (optional, uses latest if not provided)"},
                    "wait_for": {"type": "string", "description": "CSS selector or text to wait for before scraping (optional)"},
                    "wait_timeout": {"type": "number", "description": "Timeout in seconds to wait for elements (default: 30)", "default": 30}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="execute_javascript",
            description="Execute JavaScript code on a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to execute JavaScript on"},
                    "code": {"type": "string", "description": "JavaScript code to execute"},
                    "wait_for_result": {"type": "boolean", "description": "Wait for and return the result (default: true)", "default": True}
                },
                "required": ["url", "code"]
            }
        ),
        types.Tool(
            name="click_element",
            description="Click on an element on a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the page"},
                    "selector": {"type": "string", "description": "CSS selector of element to click (optional)"},
                    "text": {"type": "string", "description": "Text of element to click (optional)"},
                    "wait_after_click": {"type": "number", "description": "Seconds to wait after clicking (default: 1)", "default": 1}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="fill_form",
            description="Fill out form fields on a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the page with the form"},
                    "fields": {"type": "object", "description": "Dictionary of field selectors/names and their values"},
                    "submit": {"type": "boolean", "description": "Whether to submit the form after filling (default: false)", "default": False},
                    "submit_selector": {"type": "string", "description": "CSS selector for submit button (optional)"}
                },
                "required": ["url", "fields"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[types.TextContent, types.ImageContent]]:
    """Handle tool calls"""
    try:
        if name == "scrape_website":
            result_dict = await zendriver_manager.scrape_website(**arguments)
            return [types.TextContent(type="text", text=result_dict["result"])]
        
        elif name == "select_selectors":
            result = await zendriver_manager.select_selectors(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "scrape_with_selected_selectors":
            result = await zendriver_manager.scrape_with_selected_selectors(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "execute_javascript":
            result = await zendriver_manager.execute_javascript(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "click_element":
            result = await zendriver_manager.click_element(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "fill_form":
            result = await zendriver_manager.fill_form(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main entry point for the MCP server"""
    try:
        # Run the server using stdio transport
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="zendriver-scraper",
                    server_version="2.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await zendriver_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())