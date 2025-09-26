"""
Entity matching module for Refine-EA.

This module provides functionality for matching entities between knowledge graphs
using LLM-based reasoning and attribute comparison.
"""

from .entity_matcher import EntityMatcher
from .candidate_selector import CandidateSelector
from .attribute_extractor import AttributeExtractor

__all__ = [
    "EntityMatcher",
    "CandidateSelector", 
    "AttributeExtractor"
] 