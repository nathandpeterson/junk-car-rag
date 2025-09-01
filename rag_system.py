from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from model_manager import ModelManager
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
    """RAG system with flexible model switching"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 preferred_model: str = "llama3.2",
                 fallback_models: List[str] = None):
        
        print("🏁 Initializing 24 Hours of Lemons Virtual Inspector...")
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        self.preferred_model = preferred_model
        
        # Set up fallback order
        if fallback_models is None:
            fallback_models = ["llama3.2", "llama3.1", "gpt-3.5-turbo", "mistral"]
        
        self.model_manager.set_fallback_order(fallback_models)
        
        # Check available models
        available = self.model_manager.get_available_models()
        print(f"✅ Available models: {', '.join(available) if available else 'None'}")
        
        if not available:
            print("⚠️  No models available. Running in retrieval-only mode.")
            self.use_llm = False
        else:
            self.use_llm = True
            print(f"🤖 Preferred model: {preferred_model}")
        
        # Load chunks and embeddings (existing code)
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()
        
        print(f"✅ Loaded {len(self.chunks)} rule chunks")
        print("🏁 Virtual Inspector ready!\n")
    
    def _load_chunks(self):
        """Load rule chunks (same as before)"""
        try:
            import json
            with open('rule_chunks.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Chunks file not found. Run rule_chunker.py first!")
    
    def _load_embeddings(self):
        """Load embeddings (same as before)"""
        try:
            import numpy as np
            return np.load('rule_embeddings.npy')
        except FileNotFoundError:
            raise FileNotFoundError("Embeddings file not found. Run rule_chunker.py first!")
    
    def generate_answer_with_flexible_model(self, 
                                          query: str, 
                                          context: str,
                                          preferred_model: Optional[str] = None) -> Dict:
        """Generate answer using flexible model selection"""
        
        if not self.use_llm:
            return {"response": "LLM not available", "success": False}
        
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
        
        # Use preferred model or fallback
        model_to_use = preferred_model or self.preferred_model
        result = self.model_manager.generate_response(messages, model_to_use)
        
        return result

# Example usage and configuration
if __name__ == "__main__":
    # Example: Customize model preferences
    inspector = LemonsVirtualInspector(
        preferred_model="llama3.2",  # Try Ollama first
        fallback_models=["llama3.2", "gpt-3.5-turbo"]  # Fallback order
    )
    
    # Test switching models
    test_query = "Our team would like to put in a stronger transmission. Would this be allowed?"
    
    # Try with different models
    models_to_test = ["llama3.2", "gpt-3.5-turbo"]
    
    for model in models_to_test:
        print(f"\n--- Testing with {model} ---")
        
        # This would be integrated into your existing ask() method
        result = inspector.generate_answer_with_flexible_model(
            query=test_query,
            context="Rule 4.1: Total Investment in Vehicle Can Not Exceed $500",
            preferred_model=model
        )
        
        if result["success"]:
            print(f"✅ {result['model_used']}: {result['response']}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n🔍 Available models: {inspector.model_manager.get_available_models()}")