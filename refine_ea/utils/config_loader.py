"""
Configuration loader utilities for RefinEA.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    config_file = Path(config_path)
    
    try:
        # Create directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to: {config_path}")
    
    except Exception as e:
        raise RuntimeError(f"Error saving configuration: {e}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    def _merge_dict(target: Dict[str, Any], source: Dict[str, Any]):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                _merge_dict(target[key], value)
            else:
                target[key] = value
    
    _merge_dict(merged, override_config)
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_sections = ['model', 'generation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model configuration
    model_config = config.get('model', {})
    if 'name' not in model_config:
        raise ValueError("Model name is required in configuration")
    
    # Validate generation configuration
    generation_config = config.get('generation', {})
    required_gen_params = ['max_length', 'temperature']
    for param in required_gen_params:
        if param not in generation_config:
            raise ValueError(f"Missing required generation parameter: {param}")
    
    return True 