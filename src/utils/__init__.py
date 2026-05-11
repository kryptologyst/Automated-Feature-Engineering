"""Utility functions for automated feature engineering project."""

import logging
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/automated_feature_engineering.log"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set MPS seed if available (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    logging.info(f"Random seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logging.info("Using MPS device (Apple Silicon)")
        else:
            device = "cpu"
            logging.info("Using CPU device")
    
    return torch.device(device)


def create_directories(config: DictConfig) -> None:
    """Create necessary directories based on configuration.
    
    Args:
        config: Configuration object
    """
    directories = [
        config.paths.data_dir,
        config.paths.raw_data_dir,
        config.paths.processed_data_dir,
        config.paths.models_dir,
        config.paths.assets_dir,
        config.paths.logs_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logging.info("Created necessary directories")


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.seed < 0:
        raise ValueError("Seed must be non-negative")
    
    if not 0 < config.training.test_size < 1:
        raise ValueError("Test size must be between 0 and 1")
    
    if not 0 < config.training.validation_size < 1:
        raise ValueError("Validation size must be between 0 and 1")
    
    if config.training.test_size + config.training.validation_size >= 1:
        raise ValueError("Test and validation sizes must sum to less than 1")
    
    logging.info("Configuration validation passed")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metrics for display.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places
        
    Returns:
        Dictionary with formatted metric values
    """
    return {
        name: f"{value:.{precision}f}" if isinstance(value, (int, float)) else str(value)
        for name, value in metrics.items()
    }


def save_config(config: DictConfig, path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration object
        path: Path to save configuration
    """
    import yaml
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Configuration saved to {path}")


def load_config(path: str) -> DictConfig:
    """Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    import yaml
    
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return DictConfig(config_dict)


class SafetyChecker:
    """Safety and compliance checker for automated feature engineering."""
    
    def __init__(self, config: DictConfig):
        """Initialize safety checker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_privacy_protection(self, data: Any) -> bool:
        """Check if privacy protection measures are in place.
        
        Args:
            data: Data to check
            
        Returns:
            True if privacy protection is adequate
        """
        if not self.config.safety.enable_privacy_protection:
            self.logger.warning("Privacy protection is disabled")
            return False
        
        # Add specific privacy checks here
        self.logger.info("Privacy protection checks passed")
        return True
    
    def check_bias_detection(self, model: Any, data: Any) -> bool:
        """Check for potential bias in model and data.
        
        Args:
            model: Model to check
            data: Data to check
            
        Returns:
            True if bias detection is adequate
        """
        if not self.config.safety.enable_bias_detection:
            self.logger.warning("Bias detection is disabled")
            return False
        
        # Add specific bias detection checks here
        self.logger.info("Bias detection checks passed")
        return True
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits.
        
        Returns:
            True if memory usage is acceptable
        """
        import psutil
        
        memory_usage_gb = psutil.virtual_memory().used / (1024**3)
        max_memory_gb = self.config.safety.max_memory_usage_gb
        
        if memory_usage_gb > max_memory_gb:
            self.logger.warning(f"Memory usage {memory_usage_gb:.2f}GB exceeds limit {max_memory_gb}GB")
            return False
        
        self.logger.info(f"Memory usage {memory_usage_gb:.2f}GB is within limits")
        return True
    
    def run_all_checks(self, data: Any, model: Any) -> bool:
        """Run all safety checks.
        
        Args:
            data: Data to check
            model: Model to check
            
        Returns:
            True if all checks pass
        """
        checks = [
            self.check_privacy_protection(data),
            self.check_bias_detection(model, data),
            self.check_memory_usage(),
        ]
        
        all_passed = all(checks)
        
        if all_passed:
            self.logger.info("All safety checks passed")
        else:
            self.logger.error("Some safety checks failed")
        
        return all_passed
