import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class RuleChunk:
    """A single rule chunk ready for RAG"""
    chunk_id: str
    rule_number: str
    title: str
    content: str
    full_text: str  # Complete text for LLM context
    
    # Metadata for retrieval
    section_name: str
    section_id: str
    hierarchy_level: int
    parent_rules: List[str]
    children_rules: List[str]
    cross_references: List[str]
    
    # Keywords for hybrid search
    keywords: List[str]
    
    # Vector embedding (will be populated later)
    embedding: Optional[np.ndarray] = None
    
    # Content statistics
    char_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.full_text)
        if self.word_count == 0:
            self.word_count = len(self.full_text.split())

class RuleChunker:
    """Create optimized chunks from parsed rules for RAG"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load sentence transformer for embeddings
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Keywords that indicate important rule content
        self.important_keywords = [
            # Budget related
            'budget', 'cost', 'price', 'dollar', '$', 'expense', 'spend',
            # Safety related  
            'safety', 'helmet', 'cage', 'harness', 'fire', 'extinguisher',
            # Technical
            'engine', 'transmission', 'brake', 'suspension', 'tire', 'wheel',
            # Competition
            'penalty', 'disqualification', 'inspection', 'tech', 'scrutineering',
            # Actions
            'must', 'shall', 'required', 'mandatory', 'prohibited', 'allowed', 'permitted'
        ]
    
    def create_chunks_from_parsed_rules(self, 
                                      parsed_rules_file: str = 'parsed_rules.json',
                                      sections_file: str = 'lemons_rules.json') -> List[RuleChunk]:
        """Create rule chunks from parsed rules"""
        print(f"Creating chunks from {parsed_rules_file}")
        
        # Load parsed rules
        with open(parsed_rules_file, 'r') as f:
            parsed_rules = json.load(f)
        
        # Load original sections for context
        with open(sections_file, 'r') as f:
            sections = json.load(f)
        
        chunks = []
        
        for rule_number, rule_data in parsed_rules.items():
            chunk = self._create_chunk_from_rule(rule_data, sections, parsed_rules)
            if chunk:
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} rule chunks")
        return chunks
    
    def _create_chunk_from_rule(self, 
                               rule_data: Dict, 
                               sections: Dict, 
                               all_rules: Dict) -> Optional[RuleChunk]:
        """Create a single rule chunk"""
        
        # Get section context
        section_id = rule_data['parent_section']
        section_name = sections.get(section_id, {}).get('title', f'Section {section_id}')
        
        # Build parent rules hierarchy
        parent_rules = self._get_parent_hierarchy(rule_data['rule_number'], all_rules)
        
        # Get children rules
        children_rules = rule_data.get('children', [])
        
        # Create full text for the chunk (what the LLM will see)
        full_text = self._build_full_text(rule_data, section_name, parent_rules, all_rules)
        
        # Extract keywords
        keywords = self._extract_keywords(full_text)
        
        # Generate unique chunk ID
        chunk_id = self._generate_chunk_id(rule_data['rule_number'], rule_data['content'])
        
        return RuleChunk(
            chunk_id=chunk_id,
            rule_number=rule_data['rule_number'],
            title=rule_data['title'],
            content=rule_data['content'],
            full_text=full_text,
            section_name=section_name,
            section_id=section_id,
            hierarchy_level=rule_data['hierarchy_level'],
            parent_rules=parent_rules,
            children_rules=children_rules,
            cross_references=rule_data.get('cross_references', []),
            keywords=keywords
        )
    
    def _get_parent_hierarchy(self, rule_number: str, all_rules: Dict) -> List[str]:
        """Build list of parent rules (e.g., 4.2.1 -> ['4', '4.2'])"""
        parents = []
        parts = rule_number.split('.')
        
        for i in range(1, len(parts)):
            parent_number = '.'.join(parts[:i])
            if parent_number in all_rules:
                parents.append(parent_number)
        
        return parents
    
    def _build_full_text(self, 
                        rule_data: Dict, 
                        section_name: str, 
                        parent_rules: List[str], 
                        all_rules: Dict) -> str:
        """Build the complete text that will be sent to the LLM"""
        parts = []
        
        # Section context
        parts.append(f"Section: {section_name}")
        
        # Parent rule context (for hierarchy understanding)
        if parent_rules:
            parts.append("Parent Rules:")
            for parent in parent_rules:
                if parent in all_rules:
                    parent_rule = all_rules[parent]
                    parts.append(f"  {parent}: {parent_rule['title']}")
        
        # Main rule content
        rule_header = f"Rule {rule_data['rule_number']}"
        if rule_data['title']:
            rule_header += f": {rule_data['title']}"
        
        parts.append(f"\n{rule_header}")
        parts.append(rule_data['content'])
        
        # Cross-references
        if rule_data.get('cross_references'):
            parts.append(f"Cross-references: {', '.join(rule_data['cross_references'])}")
        
        return '\n'.join(parts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from rule text"""
        text_lower = text.lower()
        found_keywords = []
        
        # Find important predefined keywords
        for keyword in self.important_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Extract monetary amounts
        money_matches = re.findall(r'\$\d+(?:,\d+)*', text)
        found_keywords.extend(money_matches)
        
        # Extract rule numbers mentioned
        rule_refs = re.findall(r'\d+\.\d+(?:\.\d+)*', text)
        found_keywords.extend(rule_refs)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def _generate_chunk_id(self, rule_number: str, content: str) -> str:
        """Generate unique chunk ID"""
        # Combine rule number and content hash for uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"rule_{rule_number}_{content_hash}"
    
    def generate_embeddings(self, chunks: List[RuleChunk]) -> List[RuleChunk]:
        """Generate vector embeddings for all chunks"""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Prepare texts for embedding
        texts = [chunk.full_text for chunk in chunks]
        
        # Generate embeddings in batch (more efficient)
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return chunks
    
    def save_chunks(self, chunks: List[RuleChunk], filename: str = 'rule_chunks.json'):
        """Save chunks to JSON (without embeddings - too large)"""
        chunks_data = []
        
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            # Remove embedding (too large for JSON, save separately)
            chunk_dict['embedding'] = None
            chunks_data.append(chunk_dict)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(chunks)} chunks to {filename}")
    
    def save_embeddings(self, chunks: List[RuleChunk], filename: str = 'rule_embeddings.npy'):
        """Save embeddings separately as numpy array"""
        embeddings = np.array([chunk.embedding for chunk in chunks])
        np.save(filename, embeddings)
        print(f"✅ Saved embeddings to {filename}")
    
    def analyze_chunks(self, chunks: List[RuleChunk]):
        """Analyze chunk statistics"""
        print(f"\n=== Chunk Analysis ===")
        print(f"Total chunks: {len(chunks)}")
        
        # Size statistics
        char_counts = [chunk.char_count for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]
        
        print(f"Character count: avg={np.mean(char_counts):.0f}, min={min(char_counts)}, max={max(char_counts)}")
        print(f"Word count: avg={np.mean(word_counts):.0f}, min={min(word_counts)}, max={max(word_counts)}")
        
        # Hierarchy distribution
        hierarchy_levels = {}
        for chunk in chunks:
            level = chunk.hierarchy_level
            hierarchy_levels[level] = hierarchy_levels.get(level, 0) + 1
        
        print(f"Hierarchy distribution:")
        for level in sorted(hierarchy_levels.keys()):
            print(f"  Level {level}: {hierarchy_levels[level]} rules")
        
        # Section distribution
        sections = {}
        for chunk in chunks:
            section = chunk.section_name
            sections[section] = sections.get(section, 0) + 1
        
        print(f"Section distribution:")
        for section, count in sections.items():
            print(f"  {section}: {count} rules")
        
        # Show sample chunks
        print(f"\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  {chunk.rule_number}: {chunk.title}")
            print(f"    Keywords: {chunk.keywords[:5]}")
            print(f"    Size: {chunk.word_count} words, {chunk.char_count} chars")
    
    def test_similarity_search(self, chunks: List[RuleChunk], query: str, top_k: int = 3):
        """Test semantic similarity search"""
        if not chunks or chunks[0].embedding is None:
            print("❌ No embeddings found. Run generate_embeddings first.")
            return
        
        print(f"\n=== Testing Query: '{query}' ===")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            similarities.append((similarity, chunk))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Show top results
        print(f"Top {top_k} results:")
        for i, (score, chunk) in enumerate(similarities[:top_k]):
            print(f"  {i+1}. Rule {chunk.rule_number} (score: {score:.3f})")
            print(f"      {chunk.title}")
            print(f"      Keywords: {chunk.keywords[:5]}")
            print(f"      Preview: {chunk.content[:100]}...")

# Example usage
if __name__ == "__main__":
    chunker = RuleChunker()
    
    try:
        # Create chunks from parsed rules
        chunks = chunker.create_chunks_from_parsed_rules()
        
        # Analyze chunk statistics
        chunker.analyze_chunks(chunks)
        
        # Generate embeddings
        chunks = chunker.generate_embeddings(chunks)
        
        # Save chunks and embeddings
        chunker.save_chunks(chunks)
        chunker.save_embeddings(chunks)
        
        # Test some example queries
        test_queries = [
            "Can I upgrade my transmission?",
            "What's the budget limit for my car?", 
            "Safety requirements for roll cage",
            "Penalty for failing inspection"
        ]
        
        for query in test_queries:
            chunker.test_similarity_search(chunks, query)
        
        print(f"\n✅ Chunking complete! Ready for RAG system.")
        
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        print("Make sure to run the web rule parser first!")
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        import traceback
        traceback.print_exc()