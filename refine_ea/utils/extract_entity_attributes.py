#!/usr/bin/env python3
"""
Script to extract entity attributes from knowledge graph files.
Handles the conversion from entity IDs (e1, e2, etc.) to numerical indices (0, 1, etc.).
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List


def extract_entity_id_number(entity_id: str) -> int:
    """
    Extract the numerical part from entity ID.
    
    Args:
        entity_id: Entity ID like "e1", "e364", etc.
        
    Returns:
        The numerical index (0-based)
    """
    if not entity_id.startswith('e'):
        raise ValueError(f"Invalid entity ID format: {entity_id}")
    
    try:
        # Extract number after 'e' and convert to 0-based index
        number = int(entity_id[1:])
        return number - 1  # Convert to 0-based indexing
    except ValueError:
        raise ValueError(f"Invalid entity ID format: {entity_id}")


def extract_entity_attributes(kg_file_path: str) -> Dict[str, Any]:
    """
    Extract entity attributes from a knowledge graph file.
    
    Args:
        kg_file_path: Path to the knowledge graph JSON file
        
    Returns:
        Dictionary mapping entity indices to their attributes
    """
    # Read the knowledge graph file
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    entity_attributes = {}
    
    # Process each entity
    for entity in kg_data.get('entities', []):
        entity_id = entity.get('id')
        if not entity_id:
            continue
            
        # Convert entity ID to numerical index
        try:
            entity_index = extract_entity_id_number(entity_id)
        except ValueError as e:
            print(f"Warning: Skipping entity with invalid ID '{entity_id}': {e}")
            continue
        
        # Create attributes dictionary (exclude the 'id' field)
        attributes = {k: v for k, v in entity.items() if k != 'id'}
        
        # Store with numerical index as key
        entity_attributes[str(entity_index)] = attributes
    
    return entity_attributes


def main():
    parser = argparse.ArgumentParser(
        description="Extract entity attributes from knowledge graph files"
    )
    parser.add_argument(
        "kg_file",
        help="Path to the knowledge graph JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: entity_attributes.json)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    kg_path = Path(args.kg_file)
    if not kg_path.exists():
        print(f"Error: File '{kg_path}' does not exist")
        sys.exit(1)
    
    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = kg_path.parent / "entity_attributes.json"
    
    try:
        # Extract entity attributes
        print(f"Processing knowledge graph: {kg_path}")
        entity_attributes = extract_entity_attributes(str(kg_path))
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entity_attributes, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted attributes for {len(entity_attributes)} entities")
        print(f"Output saved to: {output_path}")
        
        # Print some statistics
        print("\nEntity index range:")
        if entity_attributes:
            indices = [int(k) for k in entity_attributes.keys()]
            print(f"  Min index: {min(indices)}")
            print(f"  Max index: {max(indices)}")
        
        # Show sample of entity types
        entity_types = {}
        for entity_data in entity_attributes.values():
            entity_type = entity_data.get('type', 'Unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print("\nEntity types distribution:")
        for entity_type, count in sorted(entity_types.items()):
            print(f"  {entity_type}: {count}")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 