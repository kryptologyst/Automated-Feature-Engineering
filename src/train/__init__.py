"""Main training script for automated feature engineering."""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    setup_logging, set_seed, get_device, create_directories, 
    validate_config, SafetyChecker
)
from src.data import BostonHousingDataModule, BasicFeaturePipeline, AdvancedFeaturePipeline
from src.models import ClassicalBaseline, AutoMLModel
from src.metrics import ModelEvaluator


class AutomatedFeatureEngineeringTrainer:
    """Main trainer class for automated feature engineering."""
    
    def __init__(self, config_path: str):
        """Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config.log_level)
        
        # Validate configuration
        validate_config(self.config)
        
        # Set random seed
        set_seed(self.config.seed)
        
        # Create directories
        create_directories(self.config)
        
        # Initialize components
        self.data_module = BostonHousingDataModule(self.config)
        self.feature_pipeline = None
        self.model = None
        self.evaluator = ModelEvaluator(self.config)
        self.safety_checker = SafetyChecker(self.config)
        
        self.logger.info("Automated Feature Engineering Trainer initialized")
    
    def prepare_data(self) -> tuple:
        """Prepare and split data.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Preparing data")
        
        # Load data
        X, y = self.data_module.load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_module.split_data(X, y)
        
        # Save processed data
        self.data_module.save_processed_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_feature_pipeline(self, pipeline_type: str = "basic") -> Any:
        """Create feature engineering pipeline.
        
        Args:
            pipeline_type: Type of pipeline ("basic" or "advanced")
            
        Returns:
            Feature pipeline instance
        """
        self.logger.info(f"Creating {pipeline_type} feature pipeline")
        
        if pipeline_type == "basic":
            self.feature_pipeline = BasicFeaturePipeline(self.config)
        elif pipeline_type == "advanced":
            self.feature_pipeline = AdvancedFeaturePipeline(self.config)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return self.feature_pipeline
    
    def create_model(self, model_type: str = "classical") -> Any:
        """Create model instance.
        
        Args:
            model_type: Type of model ("classical" or "automl")
            
        Returns:
            Model instance
        """
        self.logger.info(f"Creating {model_type} model")
        
        if model_type == "classical":
            self.model = ClassicalBaseline(self.config)
        elif model_type == "automl":
            self.model = AutoMLModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        model_type: str = "classical",
        pipeline_type: str = "basic"
    ) -> Dict[str, Any]:
        """Train model with feature engineering.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            model_type: Type of model to train
            pipeline_type: Type of feature pipeline
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Training {model_type} model with {pipeline_type} pipeline")
        
        # Create feature pipeline
        feature_pipeline = self.create_feature_pipeline(pipeline_type)
        
        # Create model
        model = self.create_model(model_type)
        
        # Apply feature engineering
        if pipeline_type == "advanced":
            X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
            X_val_transformed = feature_pipeline.transform(X_val)
        else:
            X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
            X_val_transformed = feature_pipeline.transform(X_val)
        
        self.logger.info(f"Feature engineering completed. Shape: {X_train_transformed.shape}")
        
        # Safety checks
        if not self.safety_checker.run_all_checks(X_train_transformed, model):
            self.logger.warning("Safety checks failed, but continuing training")
        
        # Train model
        if model_type == "automl":
            # AutoML models handle optimization internally
            model.fit(X_train_transformed, y_train)
        else:
            # Classical models
            model.fit(X_train_transformed, y_train)
        
        # Evaluate model
        X_test = self.data_module.load_processed_data()[2]  # Get test features
        y_test = self.data_module.load_processed_data()[5]   # Get test targets
        X_test_transformed = feature_pipeline.transform(X_test)
        
        model_name = f"{model_type}_{pipeline_type}"
        results = self.evaluator.evaluate_model(
            model, X_train_transformed, X_val_transformed, X_test_transformed,
            y_train, y_val, y_test, model_name
        )
        
        # Store pipeline and model for later use
        self.feature_pipeline = feature_pipeline
        self.model = model
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation with multiple models and pipelines.
        
        Returns:
            Dictionary with comprehensive results
        """
        self.logger.info("Running comprehensive evaluation")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Define experiments
        experiments = [
            ("classical", "basic"),
            ("classical", "advanced"),
            ("automl", "basic"),
            ("automl", "advanced"),
        ]
        
        results = {}
        
        for model_type, pipeline_type in experiments:
            self.logger.info(f"Running experiment: {model_type} + {pipeline_type}")
            
            try:
                experiment_results = self.train_model(
                    X_train, X_val, y_train, y_val, model_type, pipeline_type
                )
                results[f"{model_type}_{pipeline_type}"] = experiment_results
                
            except Exception as e:
                self.logger.error(f"Experiment {model_type}_{pipeline_type} failed: {e}")
                continue
        
        # Generate comparison
        comparison_df = self.evaluator.compare_models()
        
        # Get best model
        best_model = self.evaluator.get_best_model("rmse")
        
        # Get feature importance summary
        feature_importance_df = self.evaluator.get_feature_importance_summary()
        
        comprehensive_results = {
            "experiments": results,
            "comparison": comparison_df,
            "best_model": best_model,
            "feature_importance": feature_importance_df,
            "data_info": self.data_module.get_data_info(),
        }
        
        # Save results
        self.evaluator.save_results(f"{self.config.paths.assets_dir}/evaluation_results.json")
        comparison_df.to_csv(f"{self.config.paths.assets_dir}/model_comparison.csv", index=False)
        
        if not feature_importance_df.empty:
            feature_importance_df.to_csv(f"{self.config.paths.assets_dir}/feature_importance.csv")
        
        self.logger.info("Comprehensive evaluation completed")
        self.logger.info(f"Best model: {best_model}")
        
        return comprehensive_results
    
    def run_single_experiment(
        self,
        model_type: str = "classical",
        pipeline_type: str = "basic"
    ) -> Dict[str, Any]:
        """Run a single experiment.
        
        Args:
            model_type: Type of model
            pipeline_type: Type of feature pipeline
            
        Returns:
            Dictionary with experiment results
        """
        self.logger.info(f"Running single experiment: {model_type} + {pipeline_type}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Train model
        results = self.train_model(
            X_train, X_val, y_train, y_val, model_type, pipeline_type
        )
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Feature Engineering")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default="classical",
                       choices=["classical", "automl"],
                       help="Model type")
    parser.add_argument("--pipeline", type=str, default="basic",
                       choices=["basic", "advanced"],
                       help="Feature pipeline type")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive evaluation")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AutomatedFeatureEngineeringTrainer(args.config)
    
    if args.comprehensive:
        # Run comprehensive evaluation
        results = trainer.run_comprehensive_evaluation()
        
        # Print summary
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*50)
        print(f"Best Model: {results['best_model']}")
        print("\nModel Comparison:")
        print(results['comparison'].to_string(index=False))
        
        if not results['feature_importance'].empty:
            print("\nTop 10 Features by Average Importance:")
            print(results['feature_importance'].head(10).to_string())
    
    else:
        # Run single experiment
        results = trainer.run_single_experiment(args.model, args.pipeline)
        
        # Print results
        print("\n" + "="*50)
        print(f"EXPERIMENT RESULTS: {args.model} + {args.pipeline}")
        print("="*50)
        print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
        print(f"Test R²: {results['test_metrics']['r2']:.4f}")
        print(f"Test MAE: {results['test_metrics']['mae']:.4f}")
        
        if 'feature_importance' in results:
            print("\nTop 10 Features by Importance:")
            importance_df = pd.DataFrame(
                list(results['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
