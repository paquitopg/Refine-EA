"""
Base abstract interface for LLM interactions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseLLMInterface(ABC):
    """
    Abstract base class for LLM interfaces.
    
    This class defines the interface that all LLM implementations must follow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM interface.
        
        Args:
            config: Configuration dictionary containing model and generation parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._setup_logging()
        self._load_model()
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_level))
    
    @abstractmethod
    def _load_model(self):
        """
        Load the language model and tokenizer.
        
        This method should be implemented by subclasses to load the specific model.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompt texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        pass
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with provided arguments.
        
        Args:
            template: Prompt template string
            **kwargs: Arguments to format into the template
            
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)
    
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get generation parameters from configuration.
        
        Returns:
            Dictionary of generation parameters
        """
        return self.config.get('generation', {})
    
    def cleanup(self):
        """
        Clean up resources (e.g., free GPU memory).
        """
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup() 