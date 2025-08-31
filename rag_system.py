import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from sentence_transformers import SentenceTransformer
import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import logfire

load_dotenv()

# logfire.configure()
# logfire.instrument_openai()  # instrument all OpenAI clients globally

@dataclass
class RetrievedChunk:
    """A chunk retrieved for a query"""
    chunk_id: str
    rule_number: str
    title: str
    content: str
    full_text: str
    section_name: str
    keywords: List[str]
    cross_references: List[str]
    
    # Retrieval scores
    semantic_score: float
    keyword_score: float
    combined_score: float

class LemonsVirtualInspector:
    """Complete RAG system for 24 Hours of Lemons rules"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-3.5-turbo"):
        
        print("🏁 Initializing 24 Hours of Lemons Virtual Inspector...")
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Setup OpenAI (optional - can work without it for retrieval only)
        self.llm_model = llm_model
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_llm = True
            print(f"✅ LLM enabled: {llm_model}")
        else:
            self.use_llm = False
            print("⚠️  LLM disabled (no API key). Retrieval-only mode.")
        
        # Load chunks and embeddings
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()
        
        print(f"✅ Loaded {len(self.chunks)} rule chunks")
        print("🏁 Virtual Inspector ready!\n")
    
    def _load_chunks(self, filename: str = 'rule_chunks.json') -> List[Dict]:
        """Load rule chunks"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunks file {filename} not found. Run rule_chunker.py first!")
    
    def _load_embeddings(self, filename: str = 'rule_embeddings.npy') -> np.ndarray:
        """Load precomputed embeddings"""
        try:
            return np.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file {filename} not found. Run rule_chunker.py first!")
    
    def retrieve_relevant_rules(self, 
                               query: str, 
                               top_k: int = 5,
                               semantic_weight: float = 0.7,
                               keyword_weight: float = 0.3) -> List[RetrievedChunk]:
        """Retrieve most relevant rules for a query"""
        
        print(f"🔍 Searching for: '{query}'")
        
        # 1. Semantic search
        semantic_scores = self._semantic_search(query)
        
        # 2. Keyword search
        keyword_scores = self._keyword_search(query)
        
        # 3. Rule number search (if query contains rule numbers)
        rule_number_boost = self._rule_number_search(query)
        
        # 4. Combine scores
        retrieved_chunks = []
        for i, chunk in enumerate(self.chunks):
            semantic_score = semantic_scores[i]
            keyword_score = keyword_scores[i]
            rule_boost = rule_number_boost.get(chunk['rule_number'], 0)
            
            # Combined scoring with boost for exact rule matches
            combined_score = (
                semantic_weight * semantic_score + 
                keyword_weight * keyword_score +
                rule_boost
            )
            
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk['chunk_id'],
                rule_number=chunk['rule_number'],
                title=chunk['title'],
                content=chunk['content'],
                full_text=chunk['full_text'],
                section_name=chunk['section_name'],
                keywords=chunk['keywords'],
                cross_references=chunk['cross_references'],
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=combined_score
            ))
        
        # Sort by combined score and return top k
        retrieved_chunks.sort(key=lambda x: x.combined_score, reverse=True)
        top_chunks = retrieved_chunks[:top_k]
        
        print(f"📋 Found {len(top_chunks)} relevant rules:")
        for i, chunk in enumerate(top_chunks):
            print(f"  {i+1}. Rule {chunk.rule_number} (score: {chunk.combined_score:.3f})")
            print(f"     {chunk.title}")
        
        return top_chunks
    
    def _semantic_search(self, query: str) -> np.ndarray:
        """Perform semantic similarity search"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = []
        for chunk_embedding in self.embeddings:
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _keyword_search(self, query: str) -> np.ndarray:
        """Perform keyword-based search"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scores = []
        for chunk in self.chunks:
            chunk_text = (chunk['content'] + ' ' + chunk['title']).lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_text))
            
            # Calculate word overlap
            common_words = query_words.intersection(chunk_words)
            if len(query_words) > 0:
                keyword_score = len(common_words) / len(query_words)
            else:
                keyword_score = 0
            
            # Boost for exact keyword matches in chunk keywords
            for keyword in chunk.get('keywords', []):
                if keyword.lower() in query_lower:
                    keyword_score += 0.2
            
            scores.append(keyword_score)
        
        return np.array(scores)
    
    def _rule_number_search(self, query: str) -> Dict[str, float]:
        """Boost scores for explicit rule number mentions"""
        rule_pattern = re.compile(r'\b(\d+(?:\.\d+)*)\b')
        rule_numbers = rule_pattern.findall(query)
        
        boost_scores = {}
        for rule_num in rule_numbers:
            boost_scores[rule_num] = 1.0  # Strong boost for exact rule matches
        
        return boost_scores
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_chunks: List[RetrievedChunk],
                       include_cross_refs: bool = True) -> Dict:
        """Generate answer using LLM with retrieved context"""
        
        if not self.use_llm:
            # Return retrieval-only response
            return self._format_retrieval_only_response(query, retrieved_chunks)
        
        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks, include_cross_refs)
        
        # Create prompt for LLM
        prompt = self._create_prompt(query, context)
        
        try:
            # Call OpenAI API
            response = openai.responses.create(
                model=self.llm_model,
                input=[
                    {
                        "role": "system", 
                        "content": "You are a knowledgeable 24 Hours of Lemons racing inspector. Provide accurate, helpful answers about racing rules with proper citations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
            )
            
            answer = response.output_text.strip()
            
            return {
                "query": query,
                "answer": answer,
                "retrieved_rules": [
                    {
                        "rule_number": chunk.rule_number,
                        "title": chunk.title,
                        "score": chunk.combined_score
                    } for chunk in retrieved_chunks
                ],
                "context_used": context,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.llm_model
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._format_retrieval_only_response(query, retrieved_chunks)
    
    def _format_retrieval_only_response(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> Dict:
        """Format response when LLM is not available"""
        
        answer_parts = [f"Based on the rules, here are the most relevant regulations for '{query}':\n"]
        
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            answer_parts.append(f"{i}. **Rule {chunk.rule_number}**: {chunk.title}")
            answer_parts.append(f"   {chunk.content[:200]}...")
            if chunk.cross_references:
                answer_parts.append(f"   Also see: {', '.join(chunk.cross_references)}")
            answer_parts.append("")
        
        return {
            "query": query,
            "answer": "\n".join(answer_parts),
            "retrieved_rules": [
                {
                    "rule_number": chunk.rule_number,
                    "title": chunk.title,
                    "score": chunk.combined_score
                } for chunk in retrieved_chunks
            ],
            "timestamp": datetime.now().isoformat(),
            "model_used": "retrieval_only"
        }
    
    def _build_context(self, chunks: List[RetrievedChunk], include_cross_refs: bool) -> str:
        """Build context string from retrieved chunks"""
        context_parts = ["Here are the relevant racing rules:\n"]
        
        for chunk in chunks:
            context_parts.append(f"Rule {chunk.rule_number}: {chunk.title}")
            context_parts.append(chunk.content)
            
            if include_cross_refs and chunk.cross_references:
                context_parts.append(f"Cross-references: {', '.join(chunk.cross_references)}")
            
            context_parts.append("")  # Blank line between rules
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM"""
        return f"""Answer the following question about 24 Hours of Lemons racing rules. 

Use ONLY the provided rule context to answer. Always cite specific rule numbers in your response.
Be precise and helpful. If rules conflict or are unclear, mention that.

Question: {query}

Rule Context:
{context}

Provide a clear, concise answer with proper rule citations:"""
    
    def ask(self, query: str, top_k: int = 5) -> Dict:
        """Main interface: ask a question and get an answer"""
        print(f"\n{'='*60}")
        print(f"❓ Question: {query}")
        print(f"{'='*60}")
        
        # Retrieve relevant rules
        retrieved_chunks = self.retrieve_relevant_rules(query, top_k)
        
        # Generate answer
        response = self.generate_answer(query, retrieved_chunks)
        
        # Display answer
        print(f"\n🏁 Virtual Inspector Answer:")
        print(f"{response['answer']}")
        
        print(f"\n📚 Rules Referenced:")
        for rule in response['retrieved_rules']:
            print(f"  • Rule {rule['rule_number']}: {rule['title']}")
        
        return response
    
    def batch_test(self, test_queries: List[str]) -> List[Dict]:
        """Test multiple queries for evaluation"""
        results = []
        
        print(f"🧪 Running batch test on {len(test_queries)} queries...\n")
        
        for query in test_queries:
            result = self.ask(query)
            results.append(result)
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    # Note: Set OPENAI_API_KEY environment variable for LLM responses
    # or pass api_key parameter to use GPT models
    
    try:
        print("Trying OPENAI key --- ")
        print(os.getenv("OPENAI_API_KEY"))
        inspector = LemonsVirtualInspector(
            openai_api_key=os.getenv("OPENAI_API_KEY"),  # Uncomment if you have OpenAI API key
        )
        
        # Test queries from the original requirements
        test_queries = [
            "Can I upgrade my transmission?",
            "What's the budget limit for my car?",
            "Are shifters exempt from the budget?",
            "What are the roll cage requirements?",
            "Can I use the sale of old parts to offset new part costs?",
            "What safety equipment is required?",
            "What happens if I fail tech inspection?",
        ]
        
        # Run interactive demo
        print("🏁 24 Hours of Lemons Virtual Inspector Demo")
        print("Ask questions about racing rules!\n")
        
        # Test batch queries
        results = inspector.batch_test(test_queries)
        
        print(f"\n✅ Demo complete! The Virtual Inspector answered {len(results)} questions.")
        print("\nTo use interactively, call: inspector.ask('your question here')")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run rule_chunker.py first to generate the required data files!")
    except Exception as e:
        print(f"❌ Error initializing Virtual Inspector: {e}")
        import traceback
        traceback.print_exc()