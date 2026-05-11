#!/usr/bin/env python3
"""Main script to run automated feature engineering experiments."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train import AutomatedFeatureEngineeringTrainer
from src.utils import setup_logging


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Automated Feature Engineering - Research/Educational Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --comprehensive
  python run_experiments.py --model classical --pipeline basic
  python run_experiments.py --model automl --pipeline advanced --config configs/custom.yaml
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["classical", "automl"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--pipeline", 
        type=str, 
        choices=["basic", "advanced"],
        help="Feature engineering pipeline type"
    )
    
    parser.add_argument(
        "--comprehensive", 
        action="store_true",
        help="Run comprehensive evaluation with all model/pipeline combinations"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Launch interactive Streamlit demo"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run test suite"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Safety disclaimer
    print("=" * 80)
    print("⚠️  AUTOMATED FEATURE ENGINEERING - RESEARCH/EDUCATIONAL DEMO")
    print("=" * 80)
    print("DISCLAIMER: This is a research/educational demonstration.")
    print("NOT for production decisions or control systems.")
    print("Results may vary and should be validated with domain experts.")
    print("=" * 80)
    print()
    
    try:
        if args.demo:
            # Launch Streamlit demo
            logger.info("Launching Streamlit demo...")
            import subprocess
            import sys
            
            demo_path = Path(__file__).parent / "demo" / "streamlit_demo.py"
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_path)])
            
        elif args.test:
            # Run tests
            logger.info("Running test suite...")
            import subprocess
            import sys
            
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
            
        else:
            # Initialize trainer
            logger.info(f"Initializing trainer with config: {args.config}")
            trainer = AutomatedFeatureEngineeringTrainer(args.config)
            
            if args.comprehensive:
                # Run comprehensive evaluation
                logger.info("Running comprehensive evaluation...")
                results = trainer.run_comprehensive_evaluation()
                
                # Print summary
                print("\n" + "=" * 60)
                print("COMPREHENSIVE EVALUATION RESULTS")
                print("=" * 60)
                print(f"Best Model: {results['best_model']}")
                print(f"Experiments Completed: {len(results['experiments'])}")
                print("\nModel Comparison:")
                print(results['comparison'].to_string(index=False))
                
                if not results['feature_importance'].empty:
                    print("\nTop 10 Features by Average Importance:")
                    print(results['feature_importance'].head(10).to_string())
                
                print("\n" + "=" * 60)
                print("Results saved to assets/ directory")
                print("=" * 60)
                
            else:
                # Run single experiment
                if not args.model or not args.pipeline:
                    logger.error("Both --model and --pipeline are required for single experiments")
                    parser.print_help()
                    return
                
                logger.info(f"Running single experiment: {args.model} + {args.pipeline}")
                results = trainer.run_single_experiment(args.model, args.pipeline)
                
                # Print results
                print("\n" + "=" * 60)
                print(f"EXPERIMENT RESULTS: {args.model.upper()} + {args.pipeline.upper()}")
                print("=" * 60)
                print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
                print(f"Test R²: {results['test_metrics']['r2']:.4f}")
                print(f"Test MAE: {results['test_metrics']['mae']:.4f}")
                print(f"Test MAPE: {results['test_metrics']['mape']:.2f}%")
                
                if 'feature_importance' in results:
                    print("\nTop 10 Features by Importance:")
                    importance_df = pd.DataFrame(
                        list(results['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    print(importance_df.head(10).to_string(index=False))
                
                print("\n" + "=" * 60)
                print("Experiment completed successfully!")
                print("=" * 60)
    
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        print("\n⚠️  Experiment interrupted by user")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        print(f"\n❌ Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
