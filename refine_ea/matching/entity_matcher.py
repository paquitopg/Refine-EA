#!/usr/bin/env python3
"""
Entity matcher using LLM for entity alignment tasks.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from ..utils.config_loader import load_config
from ..llm import HuggingFaceInterface, vLLMInterface, BaseLLMInterface

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of entity matching."""
    entity_id: str
    best_match_id: str
    confidence_score: float
    reasoning: str
    all_candidates: List[Dict[str, Any]]
    llm_response: str


class EntityMatcher:
    """
    Entity matcher that uses LLM for entity alignment tasks.
    
    This class takes an entity from KG1 and a list of candidate entities from KG2,
    then uses an LLM to determine the best match based on entity attributes.
    """
    
    def __init__(self, config_path: str):
        """Initialize the entity matcher."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.llm = self._initialize_llm_interface()
    
    def _initialize_llm_interface(self) -> BaseLLMInterface:
        """
        Initialize the appropriate LLM interface based on configuration.
        
        Returns:
            LLM interface instance
        """
        # Check if vLLM configuration is present
        if 'api' in self.config and 'url' in self.config['api']:
            self.logger.info("Initializing vLLM HTTP interface")
            return vLLMInterface(self.config)
        else:
            self.logger.info("Initializing HuggingFace interface")
            return HuggingFaceInterface(self.config)
    
    def match_entity(
        self, 
        entity_id: str, 
        entity_attributes: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> MatchResult:
        """
        Match an entity against a list of candidates.
        
        Args:
            entity_id: ID of the entity to match
            entity_attributes: Attributes of the entity to match
            candidates: List of candidate entities with their attributes
            
        Returns:
            MatchResult containing the best match and confidence score
        """
        # Format the prompt for entity matching
        prompt = self._format_matching_prompt(entity_attributes, candidates)
        
        # Generate response using LLM
        response = self.llm.generate(prompt, max_new_tokens=1024)
        
        # Parse the response
        best_match_id, confidence_score, reasoning = self._parse_matching_response(response)
        
        # Apply confidence threshold logic
        best_match_id, confidence_score = self._apply_confidence_threshold(
            best_match_id, confidence_score
        )
        
        return MatchResult(
            entity_id=entity_id,
            best_match_id=best_match_id,
            confidence_score=confidence_score,
            reasoning=reasoning,
            all_candidates=candidates,
            llm_response=response
        )
    
    def _format_matching_prompt(
        self, 
        entity_attributes: Dict[str, Any], 
        candidates: List[Dict[str, Any]]
    ) -> str:
        """Format the prompt for entity matching."""
        
        # Format the entity to match
        entity_text = self._format_entity(entity_attributes, "Entity to match")
        
        # Format candidates
        candidates_text = ""
        for i, candidate in enumerate(candidates):
            candidates_text += f"\nCandidate {i}:\n"
            candidates_text += self._format_entity(candidate, f"Candidate {i}")
        
        prompt = f"""You are an AI assistant that helps match entities between knowledge graphs. Given an entity and a list of candidates, determine the best match.

{entity_text}

Candidate entities:{candidates_text}

Please analyze the entity and candidates, then provide:
1. The index of the best matching candidate (0, 1, 2, etc.) OR "NO_MATCH" if none of the candidates are suitable matches
2. A confidence score between 0.0 and 1.0 (use 0.0 if NO_MATCH)
3. A brief explanation of your reasoning

Response:
Best match: [candidate_index or NO_MATCH]
Confidence: [confidence_score]
Reasoning: [explanation]"""
        
        return prompt
    
    def _format_entity(self, entity: Dict[str, Any], prefix: str = "") -> str:
        """Format an entity's attributes for the prompt."""
        text = f"{prefix}:\n"
        
        # Add type if available
        if "type" in entity:
            text += f"Type: {entity['type']}\n"
        
        # Add name if available
        if "name" in entity:
            names = entity["name"] if isinstance(entity["name"], list) else [entity["name"]]
            text += f"Name: {', '.join(names)}\n"
        
        # Add description if available
        if "description" in entity:
            descriptions = entity["description"] if isinstance(entity["description"], list) else [entity["description"]]
            text += f"Description: {', '.join(descriptions)}\n"
        
        # Add other key attributes
        key_attributes = ["foundedYear", "keyStrengths", "locationName", "category"]
        for attr in key_attributes:
            if attr in entity:
                values = entity[attr] if isinstance(entity[attr], list) else [entity[attr]]
                text += f"{attr}: {', '.join(map(str, values))}\n"
        
        return text.strip()
    
    def _parse_matching_response(self, response: str) -> Tuple[str, float, str]:
        """Parse the LLM response to extract match results."""
        try:
            # Extract best match index or NO_MATCH
            best_match_match = None
            
            # First check for NO_MATCH
            no_match_pattern = re.search(r"Best match:\s*NO_MATCH", response, re.IGNORECASE)
            if no_match_pattern:
                return "NO_MATCH", 0.0, "No suitable match found among candidates"
            
            # Then check for numeric indices
            for pattern in [
                r"Best match:\s*(\d+)",
                r"best match:\s*(\d+)",
                r"candidate\s*(\d+)",
                r"index\s*(\d+)"
            ]:
                best_match_match = re.search(pattern, response, re.IGNORECASE)
                if best_match_match:
                    break
            
            best_match_id = best_match_match.group(1) if best_match_match else "0"
            
            # Extract confidence score
            confidence_match = None
            for pattern in [
                r"Confidence:\s*([0-9]*\.?[0-9]+)",
                r"confidence:\s*([0-9]*\.?[0-9]+)",
                r"score:\s*([0-9]*\.?[0-9]+)"
            ]:
                confidence_match = re.search(pattern, response, re.IGNORECASE)
                if confidence_match:
                    break
            
            confidence_score = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract reasoning
            reasoning_match = re.search(r"Reasoning:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return best_match_id, confidence_score, reasoning
            
        except Exception as e:
            self.logger.warning(f"Failed to parse response: {e}")
            return "0", 0.5, "Failed to parse response"
    
    def _apply_confidence_threshold(self, best_match_id: str, confidence_score: float) -> Tuple[str, float]:
        """
        Apply confidence threshold logic to determine if a match should be considered valid.
        
        Args:
            best_match_id: The best match ID from LLM response
            confidence_score: The confidence score from LLM response
            
        Returns:
            Tuple of (adjusted_best_match_id, adjusted_confidence_score)
        """
        # If already NO_MATCH, return as is
        if best_match_id == "NO_MATCH":
            return best_match_id, 0.0
        
        # Get threshold from config
        threshold = self.config.get("scoring", {}).get("no_match_threshold", 0.3)
        
        # If confidence is below threshold, treat as no match
        if confidence_score < threshold:
            self.logger.info(f"Confidence {confidence_score:.3f} below threshold {threshold}, treating as NO_MATCH")
            return "NO_MATCH", 0.0
        
        return best_match_id, confidence_score
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm'):
            self.llm.cleanup() 