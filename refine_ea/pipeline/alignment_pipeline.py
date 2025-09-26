#!/usr/bin/env python3
"""
Main pipeline for entity alignment using LLM reasoning.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from ..matching import EntityMatcher, CandidateSelector, AttributeExtractor
from ..matching.entity_matcher import MatchResult


@dataclass
class AlignmentResult:
    """Result of entity alignment."""
    entity_id: str
    predicted_match: str
    confidence_score: float
    reasoning: str
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    is_no_match_prediction: bool = False


class AlignmentPipeline:
    """
    Main pipeline for entity alignment using LLM reasoning.
    
    This class orchestrates the entire alignment process:
    1. Loads entity attributes and candidates
    2. Matches entities using LLM reasoning
    3. Evaluates results against ground truth
    4. Generates alignment reports
    """
    
    def __init__(self, data_dir: str, llm_config_path: str, num_candidates: int = 10):
        """Initialize the alignment pipeline."""
        self.data_dir = Path(data_dir)
        self.llm_config_path = llm_config_path
        self.num_candidates = num_candidates
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.attribute_extractor = AttributeExtractor(data_dir)
        self.candidate_selector = CandidateSelector(data_dir)
        self.entity_matcher = EntityMatcher(llm_config_path)
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        self.logger.info(f"Alignment pipeline initialized with {num_candidates} candidates per entity")
    
    def _load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth alignment pairs."""
        ground_truth_file = self.data_dir / "ref_pairs"
        
        if not ground_truth_file.exists():
            self.logger.warning(f"Ground truth file not found: {ground_truth_file}")
            return {}
        
        ground_truth = {}
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        kg1_entity_id = parts[0]
                        kg2_entity_id = parts[1]
                        ground_truth[kg1_entity_id] = kg2_entity_id
                        
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse ground truth line: {line} - {e}")
        
        self.logger.info(f"Loaded {len(ground_truth)} ground truth pairs")
        return ground_truth
    
    def align_entity(self, entity_id: str) -> AlignmentResult:
        """
        Align a single entity.
        
        Args:
            entity_id: ID of the entity to align
            
        Returns:
            AlignmentResult containing the alignment result
        """
        # Get entity attributes
        entity_attributes = self.attribute_extractor.get_entity_attributes(entity_id, kg_id=1)
        if not entity_attributes:
            self.logger.warning(f"No attributes found for entity {entity_id}")
            return AlignmentResult(
                entity_id=entity_id,
                predicted_match="",
                confidence_score=0.0,
                reasoning="No attributes found"
            )
        
        # Get candidates
        candidates_data = self.candidate_selector.get_candidates(entity_id, max_candidates=self.num_candidates)
        if not candidates_data:
            self.logger.warning(f"No candidates found for entity {entity_id}")
            return AlignmentResult(
                entity_id=entity_id,
                predicted_match="",
                confidence_score=0.0,
                reasoning="No candidates found"
            )
        
        # Extract candidate IDs
        candidate_ids = [candidate[0] for candidate in candidates_data]
        
        # Get candidate attributes
        candidate_attributes = self.attribute_extractor.get_candidate_attributes(candidate_ids, kg_id=2)
        
        # Match entity using LLM
        match_result = self.entity_matcher.match_entity(
            entity_id=entity_id,
            entity_attributes=entity_attributes,
            candidates=candidate_attributes
        )
        
        # Get the predicted match ID
        if match_result.best_match_id == "NO_MATCH":
            predicted_match = "NO_MATCH"
        else:
            try:
                predicted_match_idx = int(match_result.best_match_id)
                predicted_match = candidate_ids[predicted_match_idx] if predicted_match_idx < len(candidate_ids) else ""
            except (ValueError, IndexError):
                predicted_match = ""
        
        # Check if prediction is correct
        ground_truth_match = self.ground_truth.get(entity_id)
        is_correct = None
        if ground_truth_match:
            is_correct = predicted_match == ground_truth_match
        
        return AlignmentResult(
            entity_id=entity_id,
            predicted_match=predicted_match,
            confidence_score=match_result.confidence_score,
            reasoning=match_result.reasoning,
            ground_truth=ground_truth_match,
            is_correct=is_correct,
            is_no_match_prediction=(predicted_match == "NO_MATCH")
        )
    
    def align_entities(self, entity_ids: Optional[List[str]] = None, max_entities: Optional[int] = None) -> List[AlignmentResult]:
        """
        Align multiple entities.
        
        Args:
            entity_ids: List of entity IDs to align (if None, use all available)
            max_entities: Maximum number of entities to process
            
        Returns:
            List of AlignmentResult objects
        """
        if entity_ids is None:
            entity_ids = self.candidate_selector.get_all_entity_ids()
        
        if max_entities:
            entity_ids = entity_ids[:max_entities]
        
        results = []
        
        for i, entity_id in enumerate(entity_ids):
            self.logger.info(f"Aligning entity {entity_id} ({i+1}/{len(entity_ids)})")
            
            try:
                result = self.align_entity(entity_id)
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i+1}/{len(entity_ids)} entities")
                    
            except Exception as e:
                self.logger.error(f"Failed to align entity {entity_id}: {e}")
                results.append(AlignmentResult(
                    entity_id=entity_id,
                    predicted_match="",
                    confidence_score=0.0,
                    reasoning=f"Error: {e}",
                    is_no_match_prediction=False
                ))
        
        return results
    
    def evaluate_results(self, results: List[AlignmentResult]) -> Dict[str, Any]:
        """
        Evaluate alignment results.
        
        Args:
            results: List of AlignmentResult objects
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total = len(results)
        correct = sum(1 for r in results if r.is_correct is True)
        incorrect = sum(1 for r in results if r.is_correct is False)
        no_ground_truth = sum(1 for r in results if r.is_correct is None)
        
        # Enhanced metrics for NO_MATCH handling
        no_match_predictions = sum(1 for r in results if r.predicted_match == "NO_MATCH")
        correct_no_matches = 0
        incorrect_no_matches = 0
        
        for r in results:
            if r.predicted_match == "NO_MATCH":
                if r.ground_truth is None:
                    # Correctly predicted no match when there is no ground truth
                    correct_no_matches += 1
                else:
                    # Incorrectly predicted no match when there is a ground truth
                    incorrect_no_matches += 1
        
        # Calculate precision and recall considering NO_MATCH as valid
        true_positives = correct
        false_positives = incorrect
        false_negatives = incorrect_no_matches
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_confidence = sum(r.confidence_score for r in results) / total if total > 0 else 0.0
        
        return {
            "total_entities": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "no_ground_truth": no_ground_truth,
            "no_match_predictions": no_match_predictions,
            "correct_no_matches": correct_no_matches,
            "incorrect_no_matches": incorrect_no_matches,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_confidence": avg_confidence,
            "num_candidates": self.num_candidates
        }
    
    def save_results(self, results: List[AlignmentResult], output_file: str = "alignment_results.json"):
        """Save alignment results to file."""
        output_path = Path(output_file)
        
        # Convert results to dictionaries
        results_dict = [asdict(result) for result in results]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'entity_matcher'):
            self.entity_matcher.cleanup() 