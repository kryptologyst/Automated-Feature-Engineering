"""Boston Housing dataset module."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


class BostonHousingDataModule:
    """Data module for Boston Housing dataset (using California Housing as modern replacement)."""
    
    def __init__(self, config: DictConfig):
        """Initialize data module.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use California Housing as Boston Housing is deprecated
        self.dataset_name = "California Housing"
        self.logger.info(f"Initializing {self.dataset_name} data module")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset.
        
        Returns:
            Tuple of (features, target)
        """
        self.logger.info("Loading California Housing dataset")
        
        # Load California Housing dataset
        california_housing = fetch_california_housing()
        
        # Create DataFrame
        X = pd.DataFrame(
            california_housing.data,
            columns=california_housing.feature_names
        )
        y = pd.Series(california_housing.target, name="MedHouseVal")
        
        # Add some missing values to simulate real-world scenario
        np.random.seed(self.config.seed)
        missing_indices = np.random.choice(
            X.index, 
            size=int(0.05 * len(X)), 
            replace=False
        )
        X.loc[missing_indices, X.columns[0]] = np.nan
        
        self.logger.info(f"Loaded dataset with shape: {X.shape}")
        self.logger.info(f"Target shape: {y.shape}")
        self.logger.info(f"Missing values: {X.isnull().sum().sum()}")
        
        return X, y
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary with dataset information
        """
        X, y = self.load_data()
        
        return {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "target_name": y.name,
            "target_range": (y.min(), y.max()),
            "missing_values": X.isnull().sum().to_dict(),
            "data_types": X.dtypes.to_dict(),
        }
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Splitting data into train/validation/test sets")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_state,
            shuffle=True
        )
        
        # Second split: train vs val
        val_size = self.config.training.validation_size / (1 - self.config.training.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.training.random_state,
            shuffle=True
        )
        
        self.logger.info(f"Train set: {X_train.shape}")
        self.logger.info(f"Validation set: {X_val.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get feature descriptions.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            "MedInc": "Median income in block group",
            "HouseAge": "Median house age in block group",
            "AveRooms": "Average number of rooms per household",
            "AveBedrms": "Average number of bedrooms per household",
            "Population": "Block group population",
            "AveOccup": "Average number of household members",
            "Latitude": "Block group latitude",
            "Longitude": "Block group longitude",
        }
    
    def save_processed_data(
        self, 
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        path: str = None
    ) -> None:
        """Save processed data to files.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
            path: Base path for saving files
        """
        if path is None:
            path = self.config.paths.processed_data_dir
        
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save features
        X_train.to_csv(f"{path}/X_train.csv", index=False)
        X_val.to_csv(f"{path}/X_val.csv", index=False)
        X_test.to_csv(f"{path}/X_test.csv", index=False)
        
        # Save targets
        y_train.to_csv(f"{path}/y_train.csv", index=False)
        y_val.to_csv(f"{path}/y_val.csv", index=False)
        y_test.to_csv(f"{path}/y_test.csv", index=False)
        
        self.logger.info(f"Processed data saved to {path}")
    
    def load_processed_data(
        self, 
        path: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load processed data from files.
        
        Args:
            path: Base path for loading files
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if path is None:
            path = self.config.paths.processed_data_dir
        
        # Load features
        X_train = pd.read_csv(f"{path}/X_train.csv")
        X_val = pd.read_csv(f"{path}/X_val.csv")
        X_test = pd.read_csv(f"{path}/X_test.csv")
        
        # Load targets
        y_train = pd.read_csv(f"{path}/y_train.csv").squeeze()
        y_val = pd.read_csv(f"{path}/y_val.csv").squeeze()
        y_test = pd.read_csv(f"{path}/y_test.csv").squeeze()
        
        self.logger.info(f"Processed data loaded from {path}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
