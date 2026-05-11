"""Visualization utilities for automated feature engineering."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FeatureEngineeringVisualizer:
    """Visualization utilities for feature engineering results."""
    
    def __init__(self, config: DictConfig):
        """Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create assets directory if it doesn't exist
        import os
        os.makedirs(self.config.paths.assets_dir, exist_ok=True)
    
    def plot_data_distribution(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        save_path: Optional[str] = None
    ) -> None:
        """Plot data distribution.
        
        Args:
            X: Feature matrix
            y: Target vector
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Distribution Analysis', fontsize=16)
        
        # Target distribution
        axes[0, 0].hist(y, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_xlabel('Target Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Feature correlation heatmap
        corr_matrix = X.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0, 1], cbar_kws={'shrink': 0.8})
        axes[0, 1].set_title('Feature Correlation Matrix')
        
        # Missing values
        missing_data = X.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_data.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Missing Values by Feature')
            axes[1, 0].set_xlabel('Features')
            axes[1, 0].set_ylabel('Missing Count')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Missing Values by Feature')
        
        # Feature distributions (first 4 features)
        n_features = min(4, len(X.columns))
        for i in range(n_features):
            axes[1, 1].hist(X.iloc[:, i].dropna(), bins=20, alpha=0.7, 
                           label=X.columns[i])
        axes[1, 1].set_title('Feature Distributions')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Data distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self, 
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison results.
        
        Args:
            comparison_df: DataFrame with model comparison results
            save_path: Path to save plot
        """
        if comparison_df.empty:
            self.logger.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        # Test RMSE comparison
        models = comparison_df['model'].tolist()
        test_rmse = comparison_df['test_rmse'].tolist()
        
        axes[0, 0].bar(models, test_rmse, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Test RMSE Comparison')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Test R² comparison
        test_r2 = comparison_df['test_r2'].tolist()
        axes[0, 1].bar(models, test_r2, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Test R² Comparison')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Multiple metrics comparison
        metrics = ['test_rmse', 'test_mae', 'test_r2']
        metric_labels = ['RMSE', 'MAE', 'R²']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = comparison_df[metric].tolist()
            axes[1, 0].bar(x + i*width, values, width, label=label)
        
        axes[1, 0].set_title('Multiple Metrics Comparison')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        
        # Performance radar chart
        self._plot_radar_chart(comparison_df, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def _plot_radar_chart(self, comparison_df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot radar chart for model comparison.
        
        Args:
            comparison_df: DataFrame with model comparison results
            ax: Matplotlib axes object
        """
        # Select metrics for radar chart
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        metric_labels = ['R²', 'RMSE', 'MAE']
        
        # Normalize metrics (R² is already 0-1, normalize RMSE and MAE)
        normalized_df = comparison_df.copy()
        for metric in ['test_rmse', 'test_mae']:
            max_val = normalized_df[metric].max()
            min_val = normalized_df[metric].min()
            normalized_df[metric] = 1 - (normalized_df[metric] - min_val) / (max_val - min_val)
        
        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for _, row in normalized_df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart')
        ax.legend()
        ax.grid(True)
    
    def plot_feature_importance(
        self, 
        feature_importance: Dict[str, float],
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        if not feature_importance:
            self.logger.warning("No feature importance data to plot")
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(features)), importances, color='lightcoral', edgecolor='black')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()  # Highest importance at top
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot prediction analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Prediction Analysis', fontsize=16)
        
        # Scatter plot: True vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('True vs Predicted')
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Prediction analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_engineering_pipeline(
        self, 
        pipeline_info: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature engineering pipeline information.
        
        Args:
            pipeline_info: Dictionary with pipeline transformation information
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pipeline flow diagram
        steps = list(pipeline_info.keys())
        n_steps = len(steps)
        
        # Create boxes for each step
        for i, step in enumerate(steps):
            step_info = pipeline_info[step]
            step_type = step_info.get('type', 'Unknown')
            
            # Create rectangle
            rect = plt.Rectangle((i, 0), 0.8, 0.6, 
                               facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            
            # Add text
            ax.text(i + 0.4, 0.3, f'{step}\n{step_type}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add arrow to next step
            if i < n_steps - 1:
                ax.arrow(i + 0.8, 0.3, 0.2, 0, head_width=0.05, 
                        head_length=0.05, fc='black', ec='black')
        
        ax.set_xlim(-0.2, n_steps)
        ax.set_ylim(-0.2, 1)
        ax.set_title('Feature Engineering Pipeline', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Pipeline diagram saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(
        self, 
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Create a comprehensive summary report.
        
        Args:
            results: Dictionary with all results
            save_path: Path to save report
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Automated Feature Engineering - Comprehensive Report', 
                    fontsize=20, fontweight='bold')
        
        # Data info
        ax1 = fig.add_subplot(gs[0, :2])
        data_info = results.get('data_info', {})
        info_text = f"""
        Dataset: {data_info.get('n_samples', 'N/A')} samples, {data_info.get('n_features', 'N/A')} features
        Target Range: {data_info.get('target_range', 'N/A')}
        Missing Values: {sum(data_info.get('missing_values', {}).values())}
        """
        ax1.text(0.1, 0.5, info_text, fontsize=12, va='center')
        ax1.set_title('Dataset Information', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Best model info
        ax2 = fig.add_subplot(gs[0, 2:])
        best_model = results.get('best_model', 'N/A')
        ax2.text(0.1, 0.5, f'Best Model: {best_model}', fontsize=12, va='center')
        ax2.set_title('Best Model', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Model comparison
        ax3 = fig.add_subplot(gs[1, :])
        comparison_df = results.get('comparison', pd.DataFrame())
        if not comparison_df.empty:
            models = comparison_df['model'].tolist()
            test_rmse = comparison_df['test_rmse'].tolist()
            bars = ax3.bar(models, test_rmse, color='skyblue', edgecolor='black')
            ax3.set_title('Model Comparison (Test RMSE)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Models')
            ax3.set_ylabel('RMSE')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, test_rmse):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Feature importance
        ax4 = fig.add_subplot(gs[2, :])
        feature_importance_df = results.get('feature_importance', pd.DataFrame())
        if not feature_importance_df.empty:
            top_features = feature_importance_df.head(10)
            features = top_features.index.tolist()
            avg_importance = top_features['average'].tolist()
            
            bars = ax4.barh(features, avg_importance, color='lightcoral', edgecolor='black')
            ax4.set_title('Top 10 Features by Average Importance', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Average Importance')
            ax4.invert_yaxis()
        
        # Summary statistics
        ax5 = fig.add_subplot(gs[3, :])
        summary_text = f"""
        Experiments Completed: {len(results.get('experiments', {}))}
        Models Evaluated: {len(results.get('comparison', pd.DataFrame()))}
        Features Analyzed: {len(results.get('feature_importance', pd.DataFrame()))}
        
        This report demonstrates automated feature engineering capabilities
        with classical and advanced AutoML approaches.
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=12, va='center')
        ax5.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Summary report saved to {save_path}")
        
        plt.show()
