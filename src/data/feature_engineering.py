"""Feature engineering pipelines for automated feature engineering."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.scaling import StandardScaler, MinMaxScaler, RobustScaler
from feature_engine.transformation import LogTransformer, PowerTransformer, YeoJohnsonTransformer
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from omegaconf import DictConfig


class BasicFeaturePipeline:
    """Basic feature engineering pipeline using Feature-engine library."""
    
    def __init__(self, config: DictConfig):
        """Initialize basic feature pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformers = {}
        self.is_fitted = False
        
        self.logger.info("Initializing Basic Feature Pipeline")
    
    def _create_transformers(self) -> Dict[str, Any]:
        """Create transformer objects based on configuration.
        
        Returns:
            Dictionary of transformers
        """
        transformers = {}
        
        # Imputation
        if self.config.feature_engineering.enable_imputation:
            transformers["imputer"] = MeanMedianImputer(
                imputation_method="mean",
                variables=None  # Apply to all numeric variables
            )
        
        # Transformations
        if self.config.feature_engineering.enable_transformations:
            # Log transformation for right-skewed features
            transformers["log_transformer"] = LogTransformer(
                variables=None,  # Will be determined automatically
                base="e"
            )
            
            # Power transformation for normalization
            transformers["power_transformer"] = PowerTransformer(
                variables=None,  # Will be determined automatically
                method="yeo-johnson"
            )
        
        # Discretization
        if self.config.feature_engineering.enable_discretization:
            transformers["discretiser"] = EqualFrequencyDiscretiser(
                q=4,
                variables=None  # Will be determined automatically
            )
        
        # Scaling
        if self.config.feature_engineering.enable_scaling:
            transformers["scaler"] = StandardScaler()
        
        # Feature selection
        transformers["drop_constant"] = DropConstantFeatures(tol=0.95)
        transformers["drop_duplicates"] = DropDuplicateFeatures()
        
        return transformers
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasicFeaturePipeline":
        """Fit the feature pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Self
        """
        self.logger.info("Fitting Basic Feature Pipeline")
        
        # Create transformers
        self.transformers = self._create_transformers()
        
        # Apply transformers sequentially
        X_transformed = X.copy()
        
        for name, transformer in self.transformers.items():
            self.logger.info(f"Applying {name}")
            X_transformed = transformer.fit_transform(X_transformed, y)
        
        self.is_fitted = True
        self.logger.info("Basic Feature Pipeline fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted pipeline.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        self.logger.info("Transforming features with Basic Feature Pipeline")
        
        X_transformed = X.copy()
        
        for name, transformer in self.transformers.items():
            X_transformed = transformer.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        # Get feature names from the last transformer
        last_transformer = list(self.transformers.values())[-1]
        if hasattr(last_transformer, "feature_names_in_"):
            return list(last_transformer.feature_names_in_)
        else:
            return list(self.transformers["scaler"].feature_names_in_)
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about transformations applied.
        
        Returns:
            Dictionary with transformation information
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        info = {}
        
        for name, transformer in self.transformers.items():
            info[name] = {
                "type": type(transformer).__name__,
                "parameters": transformer.get_params(),
            }
            
            # Add specific information for certain transformers
            if hasattr(transformer, "variables_"):
                info[name]["variables"] = transformer.variables_
            if hasattr(transformer, "feature_names_in_"):
                info[name]["feature_names"] = list(transformer.feature_names_in_)
        
        return info


class AdvancedFeaturePipeline:
    """Advanced feature engineering pipeline with automated feature generation."""
    
    def __init__(self, config: DictConfig):
        """Initialize advanced feature pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformers = {}
        self.is_fitted = False
        
        self.logger.info("Initializing Advanced Feature Pipeline")
    
    def _create_transformers(self) -> Dict[str, Any]:
        """Create advanced transformer objects.
        
        Returns:
            Dictionary of transformers
        """
        transformers = {}
        
        # Basic preprocessing
        transformers["imputer"] = MeanMedianImputer(imputation_method="median")
        transformers["drop_constant"] = DropConstantFeatures(tol=0.98)
        transformers["drop_duplicates"] = DropDuplicateFeatures()
        
        # Advanced transformations
        transformers["yeo_johnson"] = YeoJohnsonTransformer(variables=None)
        
        # Polynomial features for interaction terms
        transformers["polynomial"] = PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        )
        
        # Feature selection
        transformers["feature_selection"] = SelectKBest(
            score_func=mutual_info_regression,
            k="all"  # Will be determined dynamically
        )
        
        # Scaling
        transformers["scaler"] = RobustScaler()
        
        return transformers
    
    def _determine_feature_selection_k(self, X: pd.DataFrame, y: pd.Series) -> int:
        """Determine optimal number of features to select.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Number of features to select
        """
        n_features = X.shape[1]
        
        # Use different strategies based on dataset size
        if n_features <= 10:
            return n_features
        elif n_features <= 50:
            return min(20, n_features)
        else:
            return min(50, int(0.8 * n_features))
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AdvancedFeaturePipeline":
        """Fit the advanced feature pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector (required for advanced pipeline)
            
        Returns:
            Self
        """
        if y is None:
            raise ValueError("Advanced pipeline requires target variable")
        
        self.logger.info("Fitting Advanced Feature Pipeline")
        
        # Create transformers
        self.transformers = self._create_transformers()
        
        # Determine feature selection k
        k = self._determine_feature_selection_k(X, y)
        self.transformers["feature_selection"].k = k
        
        # Apply transformers sequentially
        X_transformed = X.copy()
        
        for name, transformer in self.transformers.items():
            self.logger.info(f"Applying {name}")
            if name == "polynomial":
                # PolynomialFeatures expects numpy array
                X_transformed = pd.DataFrame(
                    transformer.fit_transform(X_transformed, y),
                    columns=[f"poly_{i}" for i in range(transformer.n_output_features_)]
                )
            else:
                X_transformed = transformer.fit_transform(X_transformed, y)
        
        self.is_fitted = True
        self.logger.info("Advanced Feature Pipeline fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted pipeline.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        self.logger.info("Transforming features with Advanced Feature Pipeline")
        
        X_transformed = X.copy()
        
        for name, transformer in self.transformers.items():
            if name == "polynomial":
                # PolynomialFeatures expects numpy array
                X_transformed = pd.DataFrame(
                    transformer.transform(X_transformed),
                    columns=[f"poly_{i}" for i in range(transformer.n_output_features_)]
                )
            else:
                X_transformed = transformer.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features.
        
        Args:
            X: Feature matrix
            y: Target vector (required for advanced pipeline)
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        feature_selection = self.transformers["feature_selection"]
        
        if hasattr(feature_selection, "scores_"):
            feature_names = self.get_feature_names()
            scores = feature_selection.scores_
            
            return dict(zip(feature_names, scores))
        
        return {}
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        # Get feature names from feature selection
        feature_selection = self.transformers["feature_selection"]
        if hasattr(feature_selection, "feature_names_in_"):
            return list(feature_selection.feature_names_in_)
        
        return []
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about transformations applied.
        
        Returns:
            Dictionary with transformation information
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        info = {}
        
        for name, transformer in self.transformers.items():
            info[name] = {
                "type": type(transformer).__name__,
                "parameters": transformer.get_params(),
            }
            
            # Add specific information
            if hasattr(transformer, "variables_"):
                info[name]["variables"] = transformer.variables_
            if hasattr(transformer, "feature_names_in_"):
                info[name]["feature_names"] = list(transformer.feature_names_in_)
            if hasattr(transformer, "n_output_features_"):
                info[name]["n_output_features"] = transformer.n_output_features_
        
        return info
