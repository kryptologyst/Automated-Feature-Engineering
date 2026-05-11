"""Tests for automated feature engineering project."""

import pytest
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils import set_seed, get_device, SafetyChecker
from src.data import BostonHousingDataModule, BasicFeaturePipeline, AdvancedFeaturePipeline
from src.models import ClassicalBaseline, AutoMLModel
from src.metrics import MetricsCalculator, ModelEvaluator


@pytest.fixture
def config():
    """Create test configuration."""
    config_dict = {
        "seed": 42,
        "device": "cpu",
        "log_level": "WARNING",
        "paths": {
            "data_dir": "test_data",
            "raw_data_dir": "test_data/raw",
            "processed_data_dir": "test_data/processed",
            "models_dir": "test_models",
            "assets_dir": "test_assets",
            "logs_dir": "test_logs"
        },
        "training": {
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42,
            "cv_folds": 3,
            "scoring": "neg_mean_squared_error"
        },
        "feature_engineering": {
            "enable_imputation": True,
            "enable_transformations": True,
            "enable_discretization": True,
            "enable_scaling": True,
            "enable_advanced": False
        },
        "model": {
            "name": "test_model",
            "params": {
                "model_type": "linear_regression",
                "random_state": 42
            }
        },
        "evaluation": {
            "metrics": ["mse", "rmse", "mae", "r2"],
            "save_predictions": True,
            "save_feature_importance": True
        },
        "safety": {
            "enable_privacy_protection": True,
            "enable_bias_detection": True,
            "max_memory_usage_gb": 8,
            "disclaimer_required": True
        }
    }
    return DictConfig(config_dict)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Add some missing values
    X.iloc[5:10, 0] = np.nan
    
    # Create target with some relationship to features
    y = pd.Series(
        X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1,
        name="target"
    )
    
    return X, y


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are reproducible
        np.random.seed(42)
        val1 = np.random.rand()
        set_seed(42)
        val2 = np.random.rand()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert str(device) == "cpu"
        
        device = get_device("auto")
        assert device is not None
    
    def test_safety_checker(self, config):
        """Test safety checker."""
        checker = SafetyChecker(config)
        
        # Test memory check
        assert checker.check_memory_usage() is True
        
        # Test privacy check
        assert checker.check_privacy_protection(None) is True
        
        # Test bias check
        assert checker.check_bias_detection(None, None) is True


class TestDataModule:
    """Test data module."""
    
    def test_boston_housing_data_module(self, config):
        """Test Boston Housing data module."""
        data_module = BostonHousingDataModule(config)
        
        # Test data loading
        X, y = data_module.load_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)
    
    def test_data_splitting(self, config):
        """Test data splitting."""
        data_module = BostonHousingDataModule(config)
        X, y = data_module.load_data()
        
        splits = data_module.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check shapes
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        
        # Check that splits are non-empty
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0


class TestFeaturePipelines:
    """Test feature engineering pipelines."""
    
    def test_basic_feature_pipeline(self, config, sample_data):
        """Test basic feature pipeline."""
        X, y = sample_data
        pipeline = BasicFeaturePipeline(config)
        
        # Test fit_transform
        X_transformed = pipeline.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        
        # Test transform
        X_transformed_2 = pipeline.transform(X)
        assert X_transformed.equals(X_transformed_2)
        
        # Test that pipeline is fitted
        assert pipeline.is_fitted is True
    
    def test_advanced_feature_pipeline(self, config, sample_data):
        """Test advanced feature pipeline."""
        X, y = sample_data
        pipeline = AdvancedFeaturePipeline(config)
        
        # Test fit_transform
        X_transformed = pipeline.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        
        # Test transform
        X_transformed_2 = pipeline.transform(X)
        assert X_transformed.equals(X_transformed_2)
        
        # Test that pipeline is fitted
        assert pipeline.is_fitted is True


class TestModels:
    """Test model implementations."""
    
    def test_classical_baseline(self, config, sample_data):
        """Test classical baseline model."""
        X, y = sample_data
        model = ClassicalBaseline(config)
        
        # Test fitting
        model.fit(X, y)
        assert model.is_fitted is True
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
        
        # Test scoring
        score = model.score(X, y)
        assert isinstance(score, float)
    
    def test_automl_model(self, config, sample_data):
        """Test AutoML model."""
        X, y = sample_data
        model = AutoMLModel(config)
        
        # Test optimization
        optimization_results = model.optimize(X, y)
        assert "best_score" in optimization_results
        assert "best_params" in optimization_results
        
        # Test fitting
        model.fit(X, y)
        assert model.is_fitted is True
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)


class TestMetrics:
    """Test metrics calculation."""
    
    def test_metrics_calculator(self, config):
        """Test metrics calculator."""
        calculator = MetricsCalculator(config)
        
        # Create sample predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = calculator.calculate_all_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ["mse", "rmse", "mae", "r2", "mape", "smape", 
                           "median_ae", "max_error", "explained_variance", "mean_absolute_scaled_error"]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_model_evaluator(self, config, sample_data):
        """Test model evaluator."""
        X, y = sample_data
        evaluator = ModelEvaluator(config)
        
        # Create a simple model
        model = ClassicalBaseline(config)
        model.fit(X, y)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        # Evaluate model
        results = evaluator.evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, "test_model")
        
        assert "test_metrics" in results
        assert "val_metrics" in results
        assert "train_metrics" in results
        assert "predictions" in results
        
        # Test model comparison
        comparison_df = evaluator.compare_models()
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) > 0


if __name__ == "__main__":
    pytest.main([__file__])
