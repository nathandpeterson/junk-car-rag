import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

@dataclass
class AccordionRuleSection:
    """Rule section extracted from accordion"""
    rule_number: str
    title: str
    content: str
    section_id: str
    hierarchy_level: int
    parent_rule: Optional[str] = None
    children: List[str] = None
    cross_references: List[str] = None

class AccordionRuleScraper:
    """Scrape rules from accordion-style websites using Selenium"""
    
    def __init__(self):
        self.rule_patterns = [
            re.compile(r'(?:Rule\s+)?(\d+(?:\.\d+)*(?:\.[a-z])*)', re.IGNORECASE),
            re.compile(r'Section\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)*)\.\s+', re.IGNORECASE),  # "4.2.1. Some rule text"
        ]
        
    def setup_driver(self, headless: bool = True):
        """Setup Chrome WebDriver"""
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # Add user agent to avoid detection
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        try:
            driver = webdriver.Chrome(options=options)
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Please make sure you have ChromeDriver installed:")
            print("  brew install chromedriver  # macOS")
            print("  Or download from: https://chromedriver.chromium.org/")
            raise
    
    def scrape_lemons_rules(self, url: str = "https://24hoursoflemons.com/prices-rules/") -> Dict[str, AccordionRuleSection]:
        """Scrape 24 Hours of Lemons accordion rules"""
        print(f"Scraping accordion rules from: {url}")
        
        driver = self.setup_driver()
        rules = {}
        
        try:
            driver.get(url)
            print("Page loaded, looking for accordion sections...")
            
            # Wait for page to load
            WebDriverWait(driver, 10).wait(
                EC.presence_of_element_located((By.CLASS_NAME, "vc_tta-panel"))
            )
            
            # Find all accordion panels
            accordion_panels = driver.find_elements(By.CLASS_NAME, "vc_tta-panel")
            print(f"Found {len(accordion_panels)} accordion sections")
            
            for i, panel in enumerate(accordion_panels):
                try:
                    # Extract section info before clicking
                    panel_id = panel.get_attribute("id") or f"section_{i}"
                    
                    # Find the header/title
                    title_element = panel.find_element(By.CLASS_NAME, "vc_tta-panel-title")
                    section_title = title_element.text.strip()
                    
                    print(f"Processing section {i+1}: {section_title}")
                    
                    # Click to expand this section
                    title_link = title_element.find_element(By.TAG_NAME, "a")
                    driver.execute_script("arguments[0].click();", title_link)
                    
                    # Wait for content to be visible
                    time.sleep(1)
                    
                    # Extract content
                    try:
                        content_element = panel.find_element(By.CLASS_NAME, "vc_tta-panel-body")
                        section_content = content_element.text.strip()
                        
                        if section_content:
                            # Try to extract rule number from title
                            rule_number = self._extract_rule_number(section_title) or str(i+1)
                            
                            rules[rule_number] = AccordionRuleSection(
                                rule_number=rule_number,
                                title=section_title,
                                content=section_content,
                                section_id=panel_id,
                                hierarchy_level=self._calculate_hierarchy_level(rule_number)
                            )
                            
                            print(f"  ✅ Extracted {len(section_content)} characters")
                        else:
                            print(f"  ⚠️  No content found")
                    
                    except NoSuchElementException:
                        print(f"  ❌ Could not find content body")
                    
                except Exception as e:
                    print(f"  ❌ Error processing panel {i}: {e}")
                    continue
        
        except TimeoutException:
            print("❌ Timeout waiting for accordion panels to load")
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
        finally:
            driver.quit()
        
        return rules
    
    def scrape_with_requests_fallback(self, url: str) -> Dict[str, AccordionRuleSection]:
        """Fallback method using requests - try to find hidden content in HTML"""
        print(f"Trying fallback method with requests...")
        
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            rules = {}
            
            # Look for accordion panels in the HTML
            panels = soup.find_all(class_="vc_tta-panel")
            print(f"Found {len(panels)} panels in HTML")
            
            for i, panel in enumerate(panels):
                try:
                    # Extract title
                    title_element = panel.find(class_="vc_tta-panel-title")
                    if title_element:
                        section_title = title_element.get_text().strip()
                    else:
                        section_title = f"Section {i+1}"
                    
                    # Extract content (might be hidden but still in HTML)
                    content_element = panel.find(class_="vc_tta-panel-body")
                    if content_element:
                        section_content = content_element.get_text().strip()
                        
                        if section_content and len(section_content) > 50:  # Ignore very short content
                            rule_number = self._extract_rule_number(section_title) or str(i+1)
                            
                            rules[rule_number] = AccordionRuleSection(
                                rule_number=rule_number,
                                title=section_title,
                                content=section_content,
                                section_id=panel.get('id', f'section_{i}'),
                                hierarchy_level=self._calculate_hierarchy_level(rule_number)
                            )
                            
                            print(f"  ✅ {section_title}: {len(section_content)} characters")
                
                except Exception as e:
                    print(f"  ❌ Error processing panel {i}: {e}")
            
            return rules
            
        except Exception as e:
            print(f"❌ Fallback method failed: {e}")
            return {}
    
    def _extract_rule_number(self, text: str) -> Optional[str]:
        """Extract rule number from text"""
        for pattern in self.rule_patterns:
            match = pattern.search(text[:100])
            if match:
                return match.group(1)
        return None
    
    def _calculate_hierarchy_level(self, rule_number: str) -> int:
        """Calculate hierarchy level based on rule numbering"""
        if re.match(r'^\d+$', rule_number):
            return 1
        elif re.match(r'^\d+\.\d+$', rule_number):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+$', rule_number):
            return 3
        else:
            return len(rule_number.split('.'))
    
    def save_rules(self, rules: Dict[str, AccordionRuleSection], filename: str = "lemons_rules.json"):
        """Save extracted rules to JSON"""
        rules_data = {}
        
        for rule_number, rule in rules.items():
            rules_data[rule_number] = {
                'rule_number': rule.rule_number,
                'title': rule.title,
                'content': rule.content,
                'section_id': rule.section_id,
                'hierarchy_level': rule.hierarchy_level,
                'parent_rule': rule.parent_rule,
                'children': rule.children,
                'cross_references': rule.cross_references
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {len(rules)} rules to {filename}")
    
    def print_rules_summary(self, rules: Dict[str, AccordionRuleSection]):
        """Print summary of extracted rules"""
        print(f"\n=== Rules Summary ===")
        print(f"Total sections extracted: {len(rules)}")
        
        # Group by hierarchy level
        by_level = {}
        for rule in rules.values():
            level = rule.hierarchy_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(rule)
        
        for level in sorted(by_level.keys()):
            print(f"Level {level}: {len(by_level[level])} rules")
        
        # Show sample rules
        print(f"\nSample rules:")
        for i, (rule_num, rule) in enumerate(rules.items()):
            if i >= 5:
                break
            print(f"  {rule_num}: {rule.title}")
            print(f"    Content preview: {rule.content[:100]}...")

# Example usage
if __name__ == "__main__":
    scraper = AccordionRuleScraper()
    
    print("24 Hours of Lemons Rule Scraper")
    print("=" * 40)
    print("This will use Selenium to handle the accordion interface")
    print("Make sure you have ChromeDriver installed!\n")
    
    try:
        # Try Selenium method first (more reliable for accordions)
        print("Method 1: Using Selenium WebDriver...")
        rules = scraper.scrape_lemons_rules()
        
        # If Selenium didn't work or got few results, try fallback
        if len(rules) < 3:
            print(f"\nSelenium only found {len(rules)} rules. Trying fallback method...")
            fallback_rules = scraper.scrape_with_requests_fallback("https://24hoursoflemons.com/prices-rules/")
            
            # Merge results (Selenium takes priority)
            for rule_num, rule in fallback_rules.items():
                if rule_num not in rules:
                    rules[rule_num] = rule
        
        if rules:
            scraper.print_rules_summary(rules)
            scraper.save_rules(rules)
        else:
            print("❌ No rules extracted. The website structure may have changed.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nIf you're having trouble with ChromeDriver, try:")
        print("  macOS: brew install chromedriver")
        print("  Ubuntu: sudo apt-get install chromium-chromedriver")
        print("  Or download from: https://chromedriver.chromium.org/")