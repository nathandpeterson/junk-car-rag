import requests
from bs4 import BeautifulSoup
import re
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time

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
    """Scrape rules from accordion-style websites"""
    
    def __init__(self):
        self.rule_patterns = [
            re.compile(r'(?:Rule\s+)?(\d+(?:\.\d+)*(?:\.[a-z])*)', re.IGNORECASE),
            re.compile(r'Section\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)*)\.\s+', re.IGNORECASE),  # "4.2.1. Some rule text"
        ]
        
    def scrape_rules(self, url: str) -> Dict[str, AccordionRuleSection]:
        """Scrape accordion rules using requests and BeautifulSoup"""
        print(f"Scraping accordion rules from: {url}")
        
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
            print(f"❌ Scraping failed: {e}")
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
    
    def save_rules(self, rules: Dict[str, AccordionRuleSection], filename: str = "data/lemons_rules.json"):
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
    parser = argparse.ArgumentParser(description="Scrape accordion rules from a given URL")
    parser.add_argument("url", type=str, help="URL of the page to scrape rules from")
    args = parser.parse_args()

    scraper = AccordionRuleScraper()
    
    print("Accordian Rule Scraper")
    print("=" * 40)
    
    try:
        rules = scraper.scrape_rules(args.url)        
        if rules:
            scraper.print_rules_summary(rules)
            scraper.save_rules(rules)
        else:
            print("❌ No rules extracted. The website structure may have changed.")
    
    except Exception as e:
        print(f"❌ Error: {e}")