#!/usr/bin/env python3

import argparse
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

load_dotenv()

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
    semantic_score: float
    keyword_score: float
    combined_score: float

class LemonsVirtualInspector:
    """Complete RAG system for 24 Hours of Lemons rules"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 model_name: str = "llama3.2:latest",
                 provider: str = "ollama"):
        
        print("🏁 Initializing 24 Hours of Lemons Virtual Inspector...")
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Setup LLM client based on provider
        self.model_name = model_name
        self.provider = provider
        self.client = self._setup_client(provider, model_name)
        
        if self.client:
            self.use_llm = True
            print(f"✅ LLM enabled: {provider}/{model_name}")
        else:
            self.use_llm = False
            print("⚠️  LLM disabled. Retrieval-only mode.")
        
        # Load chunks and embeddings
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()
        
        print(f"✅ Loaded {len(self.chunks)} rule chunks")
        print("🏁 Virtual Inspector ready!\n")
    
    def _setup_client(self, provider: str, model_name: str) -> Optional[OpenAI]:
        """Setup the appropriate client based on provider"""
        try:
            if provider.lower() == "ollama":
                return OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )
            elif provider.lower() == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("❌ OPENAI_API_KEY not found in environment")
                    return None
                return OpenAI(api_key=api_key)
            else:
                print(f"❌ Unsupported provider: {provider}")
                return None
        except Exception as e:
            print(f"❌ Failed to setup {provider} client: {e}")
            return None
    
    def _load_chunks(self, filename: str = 'rule_chunks.json') -> List[Dict]:
        """Load rule chunks"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Chunks file {filename} not found. Run rule_chunker.py first!")
            sys.exit(1)
    
    def _load_embeddings(self, filename: str = 'rule_embeddings.npy') -> np.ndarray:
        """Load precomputed embeddings"""
        try:
            return np.load(filename)
        except FileNotFoundError:
            print(f"❌ Embeddings file {filename} not found. Run rule_chunker.py first!")
            sys.exit(1)
    
    def retrieve_relevant_rules(self, 
                               query: str, 
                               top_k: int = 5,
                               semantic_weight: float = 0.7,
                               keyword_weight: float = 0.3) -> List[RetrievedChunk]:
        """Retrieve most relevant rules for a query"""
        
        print(f"🔍 Searching for: '{query}'")
        
        # Semantic search
        semantic_scores = self._semantic_search(query)
        
        # Keyword search  
        keyword_scores = self._keyword_search(query)
        
        # Rule number search
        rule_number_boost = self._rule_number_search(query)
        
        # Combine scores
        retrieved_chunks = []
        for i, chunk in enumerate(self.chunks):
            semantic_score = semantic_scores[i]
            keyword_score = keyword_scores[i]
            rule_boost = rule_number_boost.get(chunk['rule_number'], 0)
            
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
            
            # Boost for exact keyword matches
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
            boost_scores[rule_num] = 1.0
        
        return boost_scores
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_chunks: List[RetrievedChunk],
                       include_cross_refs: bool = True) -> Dict:
        """Generate answer using LLM with retrieved context"""
        
        if not self.use_llm:
            return self._format_retrieval_only_response(query, retrieved_chunks)
        
        # Build context
        context = self._build_context(retrieved_chunks, include_cross_refs)
        
        # Create messages
        messages = [
            {
                "role": "system", 
                "content": "You are a knowledgeable 24 Hours of Lemons racing inspector. Provide accurate, helpful answers about racing rules with proper citations."
            },
            {
                "role": "user", 
                "content": f"""Answer the following question about 24 Hours of Lemons racing rules. 

Use ONLY the provided rule context to answer. Always cite specific rule numbers in your response.
Be precise and helpful. If rules conflict or are unclear, mention that.

Question: {query}

Rule Context:
{context}

Provide a clear, concise answer with proper rule citations:"""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
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
                "model_used": f"{self.provider}/{self.model_name}",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._format_retrieval_only_response(query, retrieved_chunks)
    
    def _build_context(self, chunks: List[RetrievedChunk], include_cross_refs: bool) -> str:
        """Build context string from retrieved chunks"""
        context_parts = ["Here are the relevant racing rules:\n"]
        
        for chunk in chunks:
            context_parts.append(f"Rule {chunk.rule_number}: {chunk.title}")
            context_parts.append(chunk.content)
            
            if include_cross_refs and chunk.cross_references:
                context_parts.append(f"Cross-references: {', '.join(chunk.cross_references)}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
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
            "model_used": "retrieval_only",
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    
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
        
        print(f"\n🤖 Model used: {response['model_used']}")
        
        return response

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="24 Hours of Lemons Virtual Inspector - RAG-powered rule assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question using Ollama (default)
  python rag_system.py -q "What's the budget limit?"
  
  # Use specific Ollama model  
  python rag_system.py -m llama3.1:latest -q "Can I upgrade my transmission?"
  
  # Use OpenAI GPT-4
  python rag_system.py -p openai -m gpt-4 -q "What are roll cage requirements?"
  
  # Retrieval only (no LLM)
  python rag_system.py --retrieval-only -q "Safety equipment rules"
  
  # Interactive mode
  python rag_system.py --interactive
        """
    )
    
    # Model selection
    parser.add_argument('-m', '--model', 
                       default='llama3.2:latest',
                       help='Model name (default: llama3.2:latest)')
    
    parser.add_argument('-p', '--provider', 
                       choices=['ollama', 'openai'],
                       default='ollama',
                       help='Model provider (default: ollama)')
    
    # Query options
    parser.add_argument('-q', '--query', 
                       help='Question to ask the Virtual Inspector')
    
    parser.add_argument('--interactive', 
                       action='store_true',
                       help='Start interactive chat mode')
    
    # Retrieval options
    parser.add_argument('-k', '--top-k', 
                       type=int, 
                       default=5,
                       help='Number of relevant rules to retrieve (default: 5)')
    
    parser.add_argument('--retrieval-only', 
                       action='store_true',
                       help='Skip LLM generation, show retrieved rules only')
    
    # Output options
    parser.add_argument('--json', 
                       action='store_true',
                       help='Output response as JSON')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output with retrieval scores')
    
    return parser

def interactive_mode(inspector: LemonsVirtualInspector):
    """Run interactive chat mode"""
    print("\n🏁 Interactive Virtual Inspector Mode")
    print("Ask questions about 24 Hours of Lemons rules!")
    print("Type 'quit', 'exit', or 'bye' to exit.\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Thanks for using the Virtual Inspector!")
                break
            
            if not query:
                continue
            
            response = inspector.ask(query)
            print()  # Extra space for readability
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Virtual Inspector!")
            break
        except EOFError:
            break

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        # Initialize inspector
        inspector = LemonsVirtualInspector(
            model_name=args.model,
            provider=args.provider
        )
        
        # Override LLM usage if retrieval-only requested
        if args.retrieval_only:
            inspector.use_llm = False
            print("🔍 Running in retrieval-only mode")
        
        # Interactive mode
        if args.interactive:
            interactive_mode(inspector)
            return
        
        # Single query mode
        if args.query:
            response = inspector.ask(args.query, top_k=args.top_k)
            
            # JSON output
            if args.json:
                print(json.dumps(response, indent=2))
            
            return
        
        # No query provided
        print("❌ Please provide a query with -q or use --interactive mode")
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()