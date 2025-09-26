"""
LLM integration module for Refine-EA.
Provides abstract interfaces for interacting with language models.
"""

from .base import BaseLLMInterface
from .huggingface_interface import HuggingFaceInterface
from .vllm_interface import vLLMInterface
from .entity_matcher import EntityMatcher

__all__ = [
    "BaseLLMInterface",
    "HuggingFaceInterface",
    "vLLMInterface",
    "EntityMatcher"
] 