import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ParsedRule:
    """Individual rule parsed from web content"""
    rule_number: str
    title: str
    content: str
    hierarchy_level: int
    parent_section: str
    full_rule_path: str  # e.g., "4.1.1"
    parent_rule: Optional[str] = None
    children: List[str] = None
    cross_references: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.cross_references is None:
            self.cross_references = []

class WebRuleParser:
    """Parse hierarchical rules from extracted web content"""
    
    def __init__(self):
        # Enhanced patterns for various rule formats found in racing docs
        self.rule_patterns = [
            # 4.1.1 with optional title
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+?):', re.MULTILINE),
            # 4.1.1     Title: Content (with extra spaces)
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+?):\s*(.+?)(?=^\d+\.\d+|\n\n|$)', re.MULTILINE | re.DOTALL),
            # Just the number: 4.1.1
            re.compile(r'^(\d+(?:\.\d+)*)\s+', re.MULTILINE),
        ]
        
        # Cross-reference patterns
        self.cross_ref_patterns = [
            re.compile(r'(?:Rule|rule)s?\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'(?:described in|see|refer to|as per)\s+(?:Rule|rule)s?\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
        ]
    
    def parse_extracted_rules(self, json_file: str = 'data/lemons_rules.json') -> Dict[str, ParsedRule]:
        """Parse the extracted accordion rules into individual rules"""
        print(f"Parsing rules from {json_file}...")
        
        with open(json_file, 'r') as f:
            web_sections = json.load(f)
        
        all_parsed_rules = {}
        
        for section_id, section_data in web_sections.items():
            section_title = section_data['title']
            section_content = section_data['content']
            
            print(f"\n=== Parsing Section: {section_title} ===")
            
            # Parse individual rules from this section
            rules = self._parse_section_content(section_content, section_id)
            
            print(f"Found {len(rules)} individual rules in this section")
            
            # Add to overall collection
            for rule_number, rule in rules.items():
                all_parsed_rules[rule_number] = rule
        
        # Build relationships between rules
        self._build_rule_relationships(all_parsed_rules)
        
        return all_parsed_rules
    
    def _parse_section_content(self, content: str, section_id: str) -> Dict[str, ParsedRule]:
        """Parse individual rules from a section's content"""
        rules = {}
        
        # Split content into potential rule blocks
        # Look for patterns like "4.1", "4.1.1", etc.
        rule_blocks = re.split(r'\n(?=\d+\.\d+)', content)
        
        for block in rule_blocks:
            block = block.strip()
            if not block:
                continue
            
            # Try to extract rule number and content
            parsed_rule = self._parse_rule_block(block, section_id)
            if parsed_rule:
                rules[parsed_rule.rule_number] = parsed_rule
        
        return rules
    
    def _parse_rule_block(self, block: str, section_id: str) -> Optional[ParsedRule]:
        """Parse an individual rule block"""
        lines = block.split('\n')
        first_line = lines[0].strip()
        
        # Try different patterns to extract rule number and title
        for pattern in self.rule_patterns:
            match = pattern.match(first_line)
            if match:
                if len(match.groups()) >= 2:
                    rule_number = match.group(1)
                    title = match.group(2).strip()
                else:
                    rule_number = match.group(1)
                    title = ""
                
                # Get the rest of the content
                content_lines = [first_line]
                if len(lines) > 1:
                    content_lines.extend(lines[1:])
                
                full_content = '\n'.join(content_lines).strip()
                
                # Calculate hierarchy level
                hierarchy_level = len(rule_number.split('.'))
                
                # Find cross-references
                cross_refs = self._find_cross_references(full_content)
                
                return ParsedRule(
                    rule_number=rule_number,
                    title=title,
                    content=full_content,
                    hierarchy_level=hierarchy_level,
                    parent_section=section_id,
                    full_rule_path=rule_number,
                    cross_references=cross_refs
                )
        
        return None
    
    def _find_cross_references(self, content: str) -> List[str]:
        """Find cross-references to other rules in content"""
        cross_refs = []
        
        for pattern in self.cross_ref_patterns:
            matches = pattern.findall(content)
            for match in matches:
                if match not in cross_refs:
                    cross_refs.append(match)
        
        return cross_refs
    
    def _build_rule_relationships(self, rules: Dict[str, ParsedRule]):
        """Build parent-child relationships between rules"""
        
        # Build hierarchy
        for rule_number, rule in rules.items():
            parts = rule_number.split('.')
            
            # Find parent rule
            if len(parts) > 1:
                parent_parts = parts[:-1]
                parent_rule_number = '.'.join(parent_parts)
                
                if parent_rule_number in rules:
                    rule.parent_rule = parent_rule_number
                    rules[parent_rule_number].children.append(rule_number)
    
    def get_rule_by_number(self, rules: Dict[str, ParsedRule], rule_number: str) -> Optional[ParsedRule]:
        """Get a specific rule by its number"""
        return rules.get(rule_number)
    
    def get_rules_by_section(self, rules: Dict[str, ParsedRule], section: str) -> List[ParsedRule]:
        """Get all rules from a specific section (e.g., all rules starting with '4.')"""
        section_rules = []
        for rule in rules.values():
            if rule.rule_number.startswith(section):
                section_rules.append(rule)
        return section_rules
    
    def search_rules(self, rules: Dict[str, ParsedRule], query: str) -> List[ParsedRule]:
        """Search rules by content"""
        matching_rules = []
        query_lower = query.lower()
        
        for rule in rules.values():
            if (query_lower in rule.content.lower() or 
                query_lower in rule.title.lower()):
                matching_rules.append(rule)
        
        return matching_rules
    
    def print_rule_summary(self, rules: Dict[str, ParsedRule]):
        """Print a summary of parsed rules"""
        print(f"\n=== Parsed Rules Summary ===")
        print(f"Total individual rules: {len(rules)}")
        
        # Group by hierarchy level
        by_level = {}
        for rule in rules.values():
            level = rule.hierarchy_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(rule)
        
        for level in sorted(by_level.keys()):
            print(f"Level {level}: {len(by_level[level])} rules")
        
        # Show examples from each major section
        sections = set(rule.parent_section for rule in rules.values())
        for section in sorted(sections):
            section_rules = [r for r in rules.values() if r.parent_section == section]
            print(f"\nSection {section} examples:")
            for rule in section_rules[:3]:  # Show first 3
                print(f"  {rule.rule_number}: {rule.title[:60]}...")
    
    def save_parsed_rules(self, rules: Dict[str, ParsedRule], filename: str = 'parsed_rules.json'):
        """Save parsed rules to JSON"""
        rules_data = {}
        
        for rule_number, rule in rules.items():
            rules_data[rule_number] = {
                'rule_number': rule.rule_number,
                'title': rule.title,
                'content': rule.content,
                'hierarchy_level': rule.hierarchy_level,
                'parent_section': rule.parent_section,
                'full_rule_path': rule.full_rule_path,
                'parent_rule': rule.parent_rule,
                'children': rule.children,
                'cross_references': rule.cross_references
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {len(rules)} parsed rules to {filename}")

# Example usage and testing
if __name__ == "__main__":
    parser = WebRuleParser()
    
    try:
        # Parse the extracted rules
        parsed_rules = parser.parse_extracted_rules()
        
        # Show summary
        parser.print_rule_summary(parsed_rules)
        
        # Save parsed rules
        parser.save_parsed_rules(parsed_rules)
        
        # Test some specific queries
        print(f"\n=== Testing Rule Queries ===")
        
        # Look for budget/transmission rules
        budget_rules = parser.search_rules(parsed_rules, "budget")
        print(f"Rules mentioning 'budget': {len(budget_rules)}")
        for rule in budget_rules[:3]:
            print(f"  {rule.rule_number}: {rule.title}")
        
        transmission_rules = parser.search_rules(parsed_rules, "transmission")
        print(f"Rules mentioning 'transmission': {len(transmission_rules)}")
        for rule in transmission_rules:
            print(f"  {rule.rule_number}: {rule.title}")
        
        # Look for offset rules (Rule 4.7 from your example)
        offset_rules = parser.search_rules(parsed_rules, "offset")
        print(f"Rules mentioning 'offset': {len(offset_rules)}")
        for rule in offset_rules:
            print(f"  {rule.rule_number}: {rule.title}")
            print(f"    Cross-refs: {rule.cross_references}")
        
    except FileNotFoundError:
        print("❌ lemons_rules.json not found. Please run the accordion scraper first!")
    except Exception as e:
        print(f"❌ Error parsing rules: {e}")
        import traceback
        traceback.print_exc()