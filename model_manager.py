from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import os
from enum import Enum

class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"  # For future use
    AZURE = "azure"          # For future use

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = 500
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available"""
        pass

class OpenAIClient(LLMClient):
    """OpenAI API client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url
        )
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def is_available(self) -> bool:
        try:
            # Test with a simple request
            self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            return True
        except:
            return False

class OllamaClient(LLMClient):
    """Ollama local client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.base_url or "http://localhost:11434/v1",
            api_key=config.api_key or "ollama"  # Ollama doesn't need real API key
        )
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                **self.config.extra_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Ollama error: {e}")
    
    def is_available(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                stream=False
            )
            return True
        except:
            return False

class AnthropicClient(LLMClient):
    """Anthropic Claude client (placeholder for future implementation)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Would initialize Anthropic client here
        
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError("Anthropic client not implemented yet")
    
    def is_available(self) -> bool:
        return False

class ModelManager:
    """Manages multiple model configurations and provides fallback"""
    
    def __init__(self):
        self.clients: Dict[str, LLMClient] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.fallback_order: List[str] = []
        
        # Load default configurations
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default model configurations"""
        
        # OpenAI GPT models
        self.add_model("gpt-4", ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=500
        ))
        
        self.add_model("gpt-3.5-turbo", ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500
        ))
        
        # Ollama models
        self.add_model("llama3.2", ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.2:latest",
            base_url="http://localhost:11434/v1",
            temperature=0.1
        ))
        
        self.add_model("llama3.1", ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3.1:latest",
            base_url="http://localhost:11434/v1",
            temperature=0.1
        ))
        
        # self.add_model("mistral", ModelConfig(
        #     provider=ModelProvider.OLLAMA,
        #     model_name="mistral:latest",
        #     base_url="http://localhost:11434/v1",
        #     temperature=0.1
        # ))
    
    def add_model(self, name: str, config: ModelConfig):
        """Add a model configuration"""
        self.model_configs[name] = config
        
        # Create appropriate client
        if config.provider == ModelProvider.OPENAI:
            self.clients[name] = OpenAIClient(config)
        elif config.provider == ModelProvider.OLLAMA:
            self.clients[name] = OllamaClient(config)
        elif config.provider == ModelProvider.ANTHROPIC:
            self.clients[name] = AnthropicClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def set_fallback_order(self, model_names: List[str]):
        """Set the order of models to try (first available wins)"""
        self.fallback_order = model_names
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        available = []
        for name, client in self.clients.items():
            try:
                if client.is_available():
                    available.append(name)
            except:
                continue
        return available
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         preferred_model: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using preferred model or fallback"""
        
        # Determine which models to try
        models_to_try = []
        
        if preferred_model and preferred_model in self.clients:
            models_to_try.append(preferred_model)
        
        # Add fallback models
        models_to_try.extend(self.fallback_order)
        
        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))
        
        # Try each model in order
        last_error = None
        for model_name in models_to_try:
            if model_name not in self.clients:
                continue
                
            try:
                print(f"🤖 Trying model: {model_name}")
                client = self.clients[model_name]
                response = client.generate_response(messages)
                
                return {
                    "response": response,
                    "model_used": model_name,
                    "provider": client.config.provider.value,
                    "success": True
                }
                
            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                last_error = e
                continue
        
        # All models failed
        return {
            "response": f"All models failed. Last error: {last_error}",
            "model_used": None,
            "provider": None,
            "success": False,
            "error": str(last_error)
        }