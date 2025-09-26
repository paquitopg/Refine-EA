"""
vLLM HTTP implementation of the LLM interface.
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional
import logging

from .base import BaseLLMInterface

logger = logging.getLogger(__name__)


class vLLMInterface(BaseLLMInterface):
    """
    vLLM HTTP implementation of the LLM interface.
    
    Communicates with a vLLM server via HTTP requests using the OpenAI-compatible API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vLLM interface.
        
        Args:
            config: Configuration dictionary containing API settings
        """
        self.config = config
        
        # Extract API configuration
        api_config = config.get('api', {})
        self.api_url = api_config.get('url', 'http://172.16.40.54:1444/v1/chat/completions')
        self.timeout = api_config.get('timeout', 30)
        self.max_retries = api_config.get('max_retries', 3)
        self.retry_delay = api_config.get('retry_delay', 1)
        
        # Validate required configuration
        if not self.api_url:
            raise ValueError("API URL must be specified in config")
        
        self._setup_logging()
        logger.info(f"vLLM interface initialized with API URL: {self.api_url}")
    
    def _load_model(self):
        """
        No model loading needed for HTTP interface.
        The model is loaded on the server side.
        """
        logger.info("vLLM HTTP interface - no local model loading required")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using vLLM HTTP API.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Get default generation parameters
        gen_params = self.get_generation_params()
        gen_params.update(kwargs)
        
        # Get model name from config or use default
        model_name = self.config.get('model', {}).get('name', 'default')
        
        # Prepare the request payload for OpenAI-compatible API
        payload = {
            "model": model_name,  # Use the model specified in config or default
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": gen_params.get('max_new_tokens', 256),
            "temperature": gen_params.get('temperature', 0.7),
            "top_p": gen_params.get('top_p', 0.9),
            "top_k": gen_params.get('top_k', 50),
            "do_sample": gen_params.get('do_sample', True),
            "repetition_penalty": gen_params.get('repetition_penalty', 1.1),
            "stop": gen_params.get('stop', []),
            "stream": False
        }
        
        # Make the HTTP request
        response = self._make_request(payload)
        
        # Extract the generated text from OpenAI-compatible response
        if response and 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                generated_text = choice['message']['content']
                return generated_text
            elif 'text' in choice:
                # Fallback for older API format
                generated_text = choice['text']
                # Remove the input prompt if it's included in the response
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
        else:
            logger.error(f"Unexpected response format: {response}")
            return ""
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompt texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                results.append("")
        
        return results
    
    def _make_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to vLLM server with retry logic.
        
        Args:
            payload: Request payload
            
        Returns:
            Response JSON or None if failed
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making request to {self.api_url} (attempt {attempt + 1})")
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"Failed to make request after {self.max_retries} attempts")
        return None
    
    def cleanup(self):
        """
        No cleanup needed for HTTP interface.
        """
        logger.info("vLLM HTTP interface cleanup - no resources to free") 