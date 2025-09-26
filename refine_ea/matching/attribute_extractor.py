#!/usr/bin/env python3
"""
Attribute extractor for entity alignment.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class AttributeExtractor:
    """
    Extracts and manages entity attributes from knowledge graphs.
    
    This class loads entity attributes from JSON files and provides methods
    to retrieve attributes for specific entities.
    """
    
    def __init__(self, data_dir: str):
        """Initialize the attribute extractor."""
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Load entity attributes
        self.kg1_attributes = self._load_attributes("KG1_entity_attributes.json")
        self.kg2_attributes = self._load_attributes("KG2_entity_attributes.json")
        
        self.logger.info(f"Loaded {len(self.kg1_attributes)} KG1 entities and {len(self.kg2_attributes)} KG2 entities")
    
    def _load_attributes(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load entity attributes from JSON file.
        
        Args:
            filename: Name of the JSON file containing entity attributes
            
        Returns:
            Dictionary mapping entity_id to attributes
        """
        attributes_file = self.data_dir / filename
        
        if not attributes_file.exists():
            self.logger.warning(f"Attributes file not found: {attributes_file}")
            return {}
        
        try:
            with open(attributes_file, 'r', encoding='utf-8') as f:
                attributes = json.load(f)
            
            # Convert string keys to integers if possible
            converted_attributes = {}
            for key, value in attributes.items():
                try:
                    converted_key = int(key)
                    converted_attributes[converted_key] = value
                except ValueError:
                    converted_attributes[key] = value
            
            return converted_attributes
            
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load attributes from {attributes_file}: {e}")
            return {}
    
    def get_entity_attributes(self, entity_id: str, kg_id: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get attributes for a specific entity.
        
        Args:
            entity_id: ID of the entity
            kg_id: Knowledge graph ID (1 or 2)
            
        Returns:
            Dictionary of entity attributes or None if not found
        """
        try:
            # Convert entity_id to int if possible
            if isinstance(entity_id, str) and entity_id.isdigit():
                entity_id = int(entity_id)
        except ValueError:
            pass
        
        if kg_id == 1:
            return self.kg1_attributes.get(entity_id)
        elif kg_id == 2:
            return self.kg2_attributes.get(entity_id)
        else:
            self.logger.error(f"Invalid KG ID: {kg_id}")
            return None
    
    def get_candidate_attributes(self, candidate_ids: List[str], kg_id: int = 2) -> List[Dict[str, Any]]:
        """
        Get attributes for multiple candidate entities.
        
        Args:
            candidate_ids: List of candidate entity IDs
            kg_id: Knowledge graph ID (1 or 2)
            
        Returns:
            List of entity attribute dictionaries
        """
        candidates = []
        
        for candidate_id in candidate_ids:
            attributes = self.get_entity_attributes(candidate_id, kg_id)
            if attributes:
                candidates.append(attributes)
            else:
                self.logger.warning(f"No attributes found for candidate {candidate_id} in KG{kg_id}")
                # Add minimal attributes if not found
                candidates.append({"id": candidate_id, "type": "Unknown"})
        
        return candidates
    
    def get_all_entity_ids(self, kg_id: int = 1) -> List[str]:
        """Get all entity IDs for a specific knowledge graph."""
        if kg_id == 1:
            return list(self.kg1_attributes.keys())
        elif kg_id == 2:
            return list(self.kg2_attributes.keys())
        else:
            self.logger.error(f"Invalid KG ID: {kg_id}")
            return []
    
    def get_entity_count(self, kg_id: int = 1) -> int:
        """Get the number of entities in a specific knowledge graph."""
        if kg_id == 1:
            return len(self.kg1_attributes)
        elif kg_id == 2:
            return len(self.kg2_attributes)
        else:
            return 0
    
    def get_entity_names(self, entity_ids: List[str], kg_id: int = 1) -> Dict[str, str]:
        """
        Get entity names for a list of entity IDs.
        
        Args:
            entity_ids: List of entity IDs
            kg_id: Knowledge graph ID (1 or 2)
            
        Returns:
            Dictionary mapping entity_id to entity name
        """
        names = {}
        
        for entity_id in entity_ids:
            attributes = self.get_entity_attributes(entity_id, kg_id)
            if attributes and "name" in attributes:
                name = attributes["name"]
                if isinstance(name, list):
                    names[str(entity_id)] = name[0] if name else "Unknown"
                else:
                    names[str(entity_id)] = name
            else:
                names[str(entity_id)] = "Unknown"
        
        return names 