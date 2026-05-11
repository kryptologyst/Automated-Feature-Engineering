"""Metrics and evaluation framework for automated feature engineering."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
from omegaconf import DictConfig


class MetricsCalculator:
    """Calculator for various regression metrics."""
    
    def __init__(self, config: DictConfig):
        """Initialize metrics calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all available metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic regression metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics["mape"] = self._calculate_mape(y_true, y_pred)
        metrics["smape"] = self._calculate_smape(y_true, y_pred)
        metrics["median_ae"] = median_absolute_error(y_true, y_pred)
        metrics["max_error"] = np.max(np.abs(y_true - y_pred))
        
        # Custom metrics
        metrics["explained_variance"] = self._calculate_explained_variance(y_true, y_pred)
        metrics["mean_absolute_scaled_error"] = self._calculate_mase(y_true, y_pred)
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            SMAPE value
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Explained variance score
        """
        var_y = np.var(y_true)
        if var_y == 0:
            return 1.0 if np.allclose(y_true, y_pred) else 0.0
        
        return 1 - np.var(y_true - y_pred) / var_y
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MASE value
        """
        if len(y_true) < 2:
            return np.inf
        
        # Calculate naive forecast error (using previous value)
        naive_error = np.mean(np.abs(np.diff(y_true)))
        
        if naive_error == 0:
            return np.inf
        
        mae = mean_absolute_error(y_true, y_pred)
        return mae / naive_error


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, config: DictConfig):
        """Initialize model evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator(config)
        self.results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Evaluate model on train, validation, and test sets.
        
        Args:
            model: Trained model
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics for each set
        train_metrics = self.metrics_calculator.calculate_all_metrics(y_train, y_train_pred)
        val_metrics = self.metrics_calculator.calculate_all_metrics(y_val, y_val_pred)
        test_metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_test_pred)
        
        # Store results
        results = {
            "model_name": model_name,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "predictions": {
                "train": {"true": y_train, "pred": y_train_pred},
                "val": {"true": y_val, "pred": y_val_pred},
                "test": {"true": y_test, "pred": y_test_pred}
            }
        }
        
        # Add feature importance if available
        if hasattr(model, "get_feature_importance"):
            feature_importance = model.get_feature_importance()
            if feature_importance:
                results["feature_importance"] = feature_importance
        
        self.results[model_name] = results
        
        # Log key metrics
        self.logger.info(f"{model_name} - Test RMSE: {test_metrics['rmse']:.4f}")
        self.logger.info(f"{model_name} - Test R²: {test_metrics['r2']:.4f}")
        self.logger.info(f"{model_name} - Test MAE: {test_metrics['mae']:.4f}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            self.logger.warning("No models evaluated yet")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {"model": model_name}
            
            # Add test metrics
            for metric, value in results["test_metrics"].items():
                row[f"test_{metric}"] = value
            
            # Add validation metrics
            for metric, value in results["val_metrics"].items():
                row[f"val_{metric}"] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by test RMSE (lower is better)
        comparison_df = comparison_df.sort_values("test_rmse")
        
        self.logger.info("Model comparison completed")
        
        return comparison_df
    
    def get_best_model(self, metric: str = "rmse") -> Optional[str]:
        """Get the best model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model, or None if no models evaluated
        """
        if not self.results:
            return None
        
        best_model = None
        best_score = float("inf") if metric in ["mse", "rmse", "mae", "mape", "smape"] else float("-inf")
        
        for model_name, results in self.results.items():
            score = results["test_metrics"].get(metric)
            if score is None:
                continue
            
            if metric in ["mse", "rmse", "mae", "mape", "smape"]:
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:  # r2, explained_variance
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model
    
    def get_model_predictions(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get predictions for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with predictions, or None if model not found
        """
        return self.results.get(model_name, {}).get("predictions")
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance summary across all models.
        
        Returns:
            DataFrame with feature importance comparison
        """
        importance_data = []
        
        for model_name, results in self.results.items():
            feature_importance = results.get("feature_importance")
            if feature_importance:
                for feature, importance in feature_importance.items():
                    importance_data.append({
                        "model": model_name,
                        "feature": feature,
                        "importance": importance
                    })
        
        if not importance_data:
            self.logger.warning("No feature importance data available")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(importance_data)
        
        # Pivot to compare across models
        pivot_df = importance_df.pivot(index="feature", columns="model", values="importance")
        
        # Fill NaN with 0
        pivot_df = pivot_df.fillna(0)
        
        # Sort by average importance
        pivot_df["average"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values("average", ascending=False)
        
        return pivot_df
    
    def save_results(self, path: str) -> None:
        """Save evaluation results to file.
        
        Args:
            path: Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for model_name, results in self.results.items():
            serializable_results[model_name] = results.copy()
            
            # Convert predictions
            for split in ["train", "val", "test"]:
                if split in serializable_results[model_name]["predictions"]:
                    pred_data = serializable_results[model_name]["predictions"][split]
                    serializable_results[model_name]["predictions"][split] = {
                        "true": pred_data["true"].tolist(),
                        "pred": pred_data["pred"].tolist()
                    }
        
        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {path}")
    
    def load_results(self, path: str) -> None:
        """Load evaluation results from file.
        
        Args:
            path: Path to load results from
        """
        import json
        
        with open(path, "r") as f:
            loaded_results = json.load(f)
        
        # Convert back to numpy arrays
        for model_name, results in loaded_results.items():
            for split in ["train", "val", "test"]:
                if split in results["predictions"]:
                    pred_data = results["predictions"][split]
                    results["predictions"][split] = {
                        "true": np.array(pred_data["true"]),
                        "pred": np.array(pred_data["pred"])
                    }
        
        self.results = loaded_results
        self.logger.info(f"Results loaded from {path}")
