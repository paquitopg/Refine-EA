"""
HuggingFace implementation of the LLM interface.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Dict, Any, List, Optional
import logging

from .base import BaseLLMInterface

logger = logging.getLogger(__name__)


class HuggingFaceInterface(BaseLLMInterface):
    """
    HuggingFace implementation of the LLM interface.
    
    Supports both causal and seq2seq models from HuggingFace.
    """
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        model_config = self.config.get('model', {})
        model_name = model_config.get('name')
        model_type = model_config.get('type', 'auto')
        device = model_config.get('device', 'auto')
        load_in_8bit = model_config.get('load_in_8bit', False)
        load_in_4bit = model_config.get('load_in_4bit', False)
        trust_remote_code = model_config.get('trust_remote_code', True)
        
        if not model_name:
            raise ValueError("Model name must be specified in config")
        
        logger.info(f"Loading model: {model_name}")
        
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if model_type == 'causal':
            model_class = AutoModelForCausalLM
        elif model_type == 'seq2seq':
            model_class = AutoModelForSeq2SeqLM
        else:
            model_class = AutoModelForCausalLM  # Default to causal
        
        # Load with quantization if specified
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = model_class.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                device_map=device
            )
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = model_class.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                device_map=device
            )
        else:
            self.model = model_class.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            self.model.to(device)
        
        logger.info(f"Model loaded successfully on device: {device}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Get default generation parameters
        gen_params = self.get_generation_params()
        gen_params.update(kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_params.get('max_length', 512)
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            # Filter out invalid parameters
            gen_kwargs = {
                'max_new_tokens': gen_params.get('max_new_tokens', 256),
                'temperature': gen_params.get('temperature', 0.7),
                'top_p': gen_params.get('top_p', 0.9),
                'top_k': gen_params.get('top_k', 50),
                'do_sample': gen_params.get('do_sample', True),
                'num_beams': gen_params.get('num_beams', 1),
                'repetition_penalty': gen_params.get('repetition_penalty', 1.1),
                'length_penalty': gen_params.get('length_penalty', 1.0),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            # Only add early_stopping if num_beams > 1
            if gen_kwargs['num_beams'] > 1:
                gen_kwargs['early_stopping'] = gen_params.get('early_stopping', True)
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompt texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        batch_size = self.config.get('performance', {}).get('batch_size', 1)
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self._generate_batch_internal(batch_prompts, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def _generate_batch_internal(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Internal method for batch generation.
        
        Args:
            prompts: List of input prompt texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        # Get default generation parameters
        gen_params = self.get_generation_params()
        gen_params.update(kwargs)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_params.get('max_length', 512)
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            # Filter out invalid parameters
            gen_kwargs = {
                'max_new_tokens': gen_params.get('max_new_tokens', 256),
                'temperature': gen_params.get('temperature', 0.7),
                'top_p': gen_params.get('top_p', 0.9),
                'top_k': gen_params.get('top_k', 50),
                'do_sample': gen_params.get('do_sample', True),
                'num_beams': gen_params.get('num_beams', 1),
                'repetition_penalty': gen_params.get('repetition_penalty', 1.1),
                'length_penalty': gen_params.get('length_penalty', 1.0),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            # Only add early_stopping if num_beams > 1
            if gen_kwargs['num_beams'] > 1:
                gen_kwargs['early_stopping'] = gen_params.get('early_stopping', True)
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompts[i]):
                generated_text = generated_text[len(prompts[i]):].strip()
            
            generated_texts.append(generated_text)
        
        return generated_texts 