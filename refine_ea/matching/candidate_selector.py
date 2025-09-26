#!/usr/bin/env python3
"""
Candidate selector for entity alignment.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CandidateSelector:
    """
    Selects and manages candidate entities for alignment.
    
    This class loads alignment candidates from files and provides methods
    to retrieve candidates for specific entities.
    """
    
    def __init__(self, data_dir: str):
        """Initialize the candidate selector."""
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Load alignment candidates
        self.candidates = self._load_candidates()
        self.logger.info(f"Loaded {len(self.candidates)} entity candidates")
    
    def _load_candidates(self) -> Dict[str, List[Tuple[str, float, int]]]:
        """
        Load alignment candidates from file.
        
        Returns:
            Dictionary mapping entity_id to list of (candidate_id, score, rank) tuples
        """
        candidates_file = self.data_dir / "alignment_candidates.txt"
        
        if not candidates_file.exists():
            self.logger.warning(f"Candidates file not found: {candidates_file}")
            return {}
        
        candidates = {}
        
        with open(candidates_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        kg1_entity_id = parts[0]
                        kg2_entity_id = parts[1]
                        similarity_score = float(parts[2])
                        rank = int(parts[3])
                        
                        if kg1_entity_id not in candidates:
                            candidates[kg1_entity_id] = []
                        
                        candidates[kg1_entity_id].append((kg2_entity_id, similarity_score, rank))
                        
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse line: {line} - {e}")
        
        # Sort candidates by rank for each entity
        for entity_id in candidates:
            candidates[entity_id].sort(key=lambda x: x[2])
        
        return candidates
    
    def get_candidates(self, entity_id: str, max_candidates: int = 10) -> List[Tuple[str, float, int]]:
        """
        Get candidates for a specific entity.
        
        Args:
            entity_id: ID of the entity to get candidates for
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (candidate_id, score, rank) tuples
        """
        if entity_id not in self.candidates:
            self.logger.warning(f"No candidates found for entity {entity_id}")
            return []
        
        return self.candidates[entity_id][:max_candidates]
    
    def get_all_entity_ids(self) -> List[str]:
        """Get all entity IDs that have candidates."""
        return list(self.candidates.keys())
    
    def get_candidate_count(self, entity_id: str) -> int:
        """Get the number of candidates for a specific entity."""
        return len(self.candidates.get(entity_id, []))
    
    def get_top_candidate(self, entity_id: str) -> Optional[Tuple[str, float, int]]:
        """Get the top-ranked candidate for a specific entity."""
        candidates = self.get_candidates(entity_id, max_candidates=1)
        return candidates[0] if candidates else None 