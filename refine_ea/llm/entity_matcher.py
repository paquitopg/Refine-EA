"""
Entity matcher using LLM for entity alignment tasks.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base import BaseLLMInterface

logger = logging.getLogger(__name__)


class EntityMatcher:
    """
    Entity matcher that uses LLM for entity alignment tasks.
    
    This class provides methods for matching entities using language models
    and extracting confidence scores from the responses.
    """
    
    def __init__(self, llm_interface: BaseLLMInterface):
        """
        Initialize the entity matcher.
        
        Args:
            llm_interface: LLM interface instance
        """
        self.llm = llm_interface
        self.config = llm_interface.config
    
    def match_entity_to_candidates(
        self, 
        entity: Dict[str, Any], 
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Match an entity to a list of candidate entities.
        
        Args:
            entity: Entity to match (with attributes)
            candidates: List of candidate entities
            
        Returns:
            Dictionary containing match results
        """
        # Format entity description
        entity_desc = self._format_entity_description(entity)
        
        # Format candidate descriptions
        candidate_descs = []
        for i, candidate in enumerate(candidates):
            candidate_desc = self._format_entity_description(candidate, candidate_id=i)
            candidate_descs.append(candidate_desc)
        
        # Create prompt
        prompt_template = self.config.get('prompts', {}).get('entity_matching', '')
        prompt = self.llm.format_prompt(
            prompt_template,
            entity_description=entity_desc,
            candidate_entities='\n'.join(candidate_descs)
        )
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Debug: print the prompt and response
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Response: {response}")
        
        # Parse response
        result = self._parse_matching_response(response, len(candidates))
        
        return {
            'entity': entity,
            'candidates': candidates,
            'response': response,
            'match_result': result
        }
    
    def compare_entities(
        self, 
        entity1: Dict[str, Any], 
        entity2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two entities for similarity.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Dictionary containing comparison results
        """
        # Format entity descriptions
        entity1_desc = self._format_entity_description(entity1, entity_id="Entity 1")
        entity2_desc = self._format_entity_description(entity2, entity_id="Entity 2")
        
        # Create prompt
        prompt_template = self.config.get('prompts', {}).get('entity_comparison', '')
        prompt = self.llm.format_prompt(
            prompt_template,
            entity1_description=entity1_desc,
            entity2_description=entity2_desc
        )
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Debug: print the prompt and response
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Response: {response}")
        
        # Parse response
        result = self._parse_comparison_response(response)
        
        return {
            'entity1': entity1,
            'entity2': entity2,
            'response': response,
            'comparison_result': result
        }
    
    def _format_entity_description(
        self, 
        entity: Dict[str, Any], 
        entity_id: Optional[str] = None,
        candidate_id: Optional[int] = None
    ) -> str:
        """
        Format entity attributes into a readable description.
        
        Args:
            entity: Entity dictionary
            entity_id: Optional entity identifier
            candidate_id: Optional candidate index
            
        Returns:
            Formatted entity description
        """
        lines = []
        
        # Add identifier
        if entity_id:
            lines.append(f"ID: {entity_id}")
        elif candidate_id is not None:
            lines.append(f"Candidate {candidate_id}:")
        
        # Add type
        if 'type' in entity:
            lines.append(f"Type: {entity['type']}")
        
        # Add name
        if 'name' in entity:
            names = entity['name']
            if isinstance(names, list):
                names = ', '.join(names)
            lines.append(f"Name: {names}")
        
        # Add description
        if 'description' in entity:
            descriptions = entity['description']
            if isinstance(descriptions, list):
                descriptions = ' '.join(descriptions)
            lines.append(f"Description: {descriptions}")
        
        # Add other attributes
        for key, value in entity.items():
            if key not in ['id', 'type', 'name', 'description']:
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                lines.append(f"{key.title()}: {value}")
        
        return '\n'.join(lines)
    
    def _parse_matching_response(self, response: str, num_candidates: int) -> Dict[str, Any]:
        """
        Parse the LLM response for entity matching.
        
        Args:
            response: LLM response text
            num_candidates: Number of candidates
            
        Returns:
            Parsed result dictionary
        """
        result = {
            'best_match': None,
            'confidence': 0.0,
            'reasoning': '',
            'all_scores': []
        }
        
        # Extract best match
        match_pattern = r'Best match:\s*\[?(\d+)\]?'
        match_match = re.search(match_pattern, response, re.IGNORECASE)
        if match_match:
            try:
                best_match_idx = int(match_match.group(1))
                if 0 <= best_match_idx < num_candidates:
                    result['best_match'] = best_match_idx
            except ValueError:
                pass
        
        # If no match found, try alternative patterns
        if result['best_match'] is None:
            # Try to find any number that could be a candidate index
            alt_pattern = r'(\d+)'
            numbers = re.findall(alt_pattern, response)
            for num_str in numbers:
                try:
                    idx = int(num_str)
                    if 0 <= idx < num_candidates:
                        result['best_match'] = idx
                        break
                except ValueError:
                    continue
        
        # Extract confidence score
        confidence_pattern = r'Confidence:\s*\[?([0-9]*\.?[0-9]+)\]?'
        confidence_match = re.search(confidence_pattern, response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                result['confidence'] = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        
        # If no confidence found, try alternative patterns
        if result['confidence'] == 0.0:
            # Try to find any decimal number between 0 and 1
            alt_confidence_pattern = r'([0-9]*\.[0-9]+|[0-9]+)'
            numbers = re.findall(alt_confidence_pattern, response)
            for num_str in numbers:
                try:
                    confidence = float(num_str)
                    if 0.0 <= confidence <= 1.0:
                        result['confidence'] = confidence
                        break
                except ValueError:
                    continue
        
        # Extract reasoning
        reasoning_pattern = r'Reasoning:\s*(.+)'
        reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        return result
    
    def _parse_comparison_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response for entity comparison.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed result dictionary
        """
        result = {
            'similarity': 0.0,
            'reasoning': ''
        }
        
        # Extract similarity score
        similarity_pattern = r'Similarity:\s*\[?([0-9]*\.?[0-9]+)\]?'
        similarity_match = re.search(similarity_pattern, response, re.IGNORECASE)
        if similarity_match:
            try:
                similarity = float(similarity_match.group(1))
                result['similarity'] = max(0.0, min(1.0, similarity))
            except ValueError:
                pass
        
        # Extract reasoning
        reasoning_pattern = r'Reasoning:\s*(.+)'
        reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        return result
    
    def batch_match_entities(
        self, 
        entity_pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Match multiple entities to their candidates in batch.
        
        Args:
            entity_pairs: List of (entity, candidates) tuples
            
        Returns:
            List of match results
        """
        prompts = []
        for entity, candidates in entity_pairs:
            entity_desc = self._format_entity_description(entity)
            candidate_descs = []
            for i, candidate in enumerate(candidates):
                candidate_desc = self._format_entity_description(candidate, candidate_id=i)
                candidate_descs.append(candidate_desc)
            
            prompt_template = self.config.get('prompts', {}).get('entity_matching', '')
            prompt = self.llm.format_prompt(
                prompt_template,
                entity_description=entity_desc,
                candidate_entities='\n'.join(candidate_descs)
            )
            prompts.append(prompt)
        
        # Generate responses in batch
        responses = self.llm.generate_batch(prompts)
        
        # Parse responses
        results = []
        for i, (entity, candidates) in enumerate(entity_pairs):
            response = responses[i]
            result = self._parse_matching_response(response, len(candidates))
            
            results.append({
                'entity': entity,
                'candidates': candidates,
                'response': response,
                'match_result': result
            })
        
        return results 