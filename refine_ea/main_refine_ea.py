#!/usr/bin/env python3
"""
Main entry point for Refine-EA entity alignment pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from refine_ea.pipeline import AlignmentPipeline


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('refine_ea.log')
        ]
    )


def main():
    """Main entry point for the Refine-EA pipeline."""
    parser = argparse.ArgumentParser(
        description="Refine-EA: Entity Alignment using LLM Reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory containing entity attributes and alignment files"
    )
    
    parser.add_argument(
        "--llm_config",
        type=str,
        required=True,
        help="Path to the LLM configuration file"
    )
    
    parser.add_argument(
        "--num_candidates",
        type=int,
        required=True,
        help="Number of candidates to consider per entity"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--max_entities",
        type=int,
        help="Maximum number of entities to process (default: all available)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    llm_config_path = Path(args.llm_config)
    if not llm_config_path.exists():
        logger.error(f"LLM config file does not exist: {llm_config_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting RefinEA Entity Alignment Pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"LLM config: {llm_config_path}")
    logger.info(f"Number of candidates: {args.num_candidates}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize pipeline
        pipeline = AlignmentPipeline(
            data_dir=str(data_dir),
            llm_config_path=str(llm_config_path),
            num_candidates=args.num_candidates
        )
        
        # Align entities
        logger.info("📊 Starting entity alignment...")
        results = pipeline.align_entities(max_entities=args.max_entities)
        
        # Print results summary
        logger.info(f"✅ Aligned {len(results)} entities")
        
        # Evaluate results
        logger.info("📈 Evaluating results...")
        metrics = pipeline.evaluate_results(results)
        
        # Print evaluation metrics
        logger.info("📊 Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save results
        output_file = output_dir / "alignment_results.json"
        pipeline.save_results(results, str(output_file))
        
        # Save evaluation metrics
        metrics_file = output_dir / "evaluation_metrics.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_file}")
        logger.info(f"📊 Metrics saved to {metrics_file}")
        
        # Print detailed results
        logger.info("\n📋 Detailed Results:")
        logger.info("-" * 50)
        
        for result in results:
            status = "✅" if result.is_correct else "❌" if result.is_correct is False else "❓"
            logger.info(f"{status} Entity {result.entity_id} -> {result.predicted_match} (confidence: {result.confidence_score:.3f})")
            if result.reasoning:
                logger.info(f"   Reasoning: {result.reasoning[:100]}...")
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logging.error("Pipeline failed", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        if 'pipeline' in locals():
            pipeline.cleanup()
        logger.info("🧹 Cleanup completed")


if __name__ == "__main__":
    main() 