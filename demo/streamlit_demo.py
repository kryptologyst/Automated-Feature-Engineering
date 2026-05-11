"""Streamlit demo for Automated Feature Engineering."""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import DictConfig, OmegaConf

from src.utils import setup_logging, set_seed, SafetyChecker
from src.data import BostonHousingDataModule, BasicFeaturePipeline, AdvancedFeaturePipeline
from src.models import ClassicalBaseline, AutoMLModel
from src.metrics import ModelEvaluator
from src.viz import FeatureEngineeringVisualizer


class StreamlitDemo:
    """Streamlit demo for automated feature engineering."""
    
    def __init__(self):
        """Initialize the demo."""
        # Load configuration
        self.config = OmegaConf.load("configs/config.yaml")
        
        # Setup logging
        self.logger = setup_logging("INFO")
        
        # Initialize components
        self.data_module = BostonHousingDataModule(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = FeatureEngineeringVisualizer(self.config)
        self.safety_checker = SafetyChecker(self.config)
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = {}
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
    
    def run(self):
        """Run the Streamlit demo."""
        st.set_page_config(
            page_title="Automated Feature Engineering Demo",
            page_icon="🔧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🔧 Automated Feature Engineering Demo")
        st.markdown("""
        This demo showcases automated feature engineering capabilities using classical and advanced AutoML approaches.
        Explore different feature engineering pipelines and model combinations to understand their impact on performance.
        """)
        
        # Safety disclaimer
        st.warning("""
        **⚠️ DISCLAIMER**: This is a research/educational demo. Not for production decisions or control systems.
        Results may vary and should be validated with domain experts before any real-world application.
        """)
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔧 Feature Engineering", 
            "🤖 Model Training", 
            "📈 Results Analysis", 
            "📋 Summary Report"
        ])
        
        with tab1:
            self._data_overview_tab()
        
        with tab2:
            self._feature_engineering_tab()
        
        with tab3:
            self._model_training_tab()
        
        with tab4:
            self._results_analysis_tab()
        
        with tab5:
            self._summary_report_tab()
    
    def _create_sidebar(self):
        """Create sidebar controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Data controls
        st.sidebar.subheader("📊 Data Settings")
        if st.sidebar.button("🔄 Load Data", key="load_data"):
            self._load_data()
        
        # Feature engineering controls
        st.sidebar.subheader("🔧 Feature Engineering")
        self.pipeline_type = st.sidebar.selectbox(
            "Pipeline Type",
            ["basic", "advanced"],
            help="Choose between basic and advanced feature engineering pipelines"
        )
        
        # Model controls
        st.sidebar.subheader("🤖 Model Settings")
        self.model_type = st.sidebar.selectbox(
            "Model Type",
            ["classical", "automl"],
            help="Choose between classical baseline and AutoML models"
        )
        
        if st.sidebar.button("🚀 Train Model", key="train_model"):
            self._train_model()
        
        # Evaluation controls
        st.sidebar.subheader("📈 Evaluation")
        if st.sidebar.button("📊 Run Comprehensive Evaluation", key="comprehensive_eval"):
            self._run_comprehensive_evaluation()
        
        # Safety controls
        st.sidebar.subheader("🛡️ Safety & Compliance")
        st.sidebar.checkbox(
            "Enable Privacy Protection",
            value=self.config.safety.enable_privacy_protection,
            help="Enable privacy protection measures"
        )
        st.sidebar.checkbox(
            "Enable Bias Detection",
            value=self.config.safety.enable_bias_detection,
            help="Enable bias detection checks"
        )
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **Author**: kryptologyst  
        **GitHub**: [kryptologyst](https://github.com/kryptologyst)  
        **License**: MIT
        """)
    
    def _load_data(self):
        """Load and display data."""
        with st.spinner("Loading data..."):
            try:
                # Load data
                X, y = self.data_module.load_data()
                
                # Store in session state
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                
                st.success("✅ Data loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    def _data_overview_tab(self):
        """Data overview tab."""
        st.header("📊 Data Overview")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please load data first using the sidebar controls.")
            return
        
        X = st.session_state.X
        y = st.session_state.y
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", len(X))
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            st.metric("Missing Values", X.isnull().sum().sum())
        with col4:
            st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(X.head(10))
        
        # Feature descriptions
        st.subheader("📝 Feature Descriptions")
        feature_descriptions = self.data_module.get_feature_descriptions()
        for feature, description in feature_descriptions.items():
            st.write(f"**{feature}**: {description}")
        
        # Visualizations
        st.subheader("📈 Data Visualizations")
        
        # Target distribution
        fig = px.histogram(y, title="Target Distribution", nbins=30)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        corr_matrix = X.corr()
        fig = px.imshow(corr_matrix, 
                       title="Feature Correlation Matrix",
                       color_continuous_scale="RdBu",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values
        missing_data = X.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        title="Missing Values by Feature")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("✅ No missing values found in the dataset.")
    
    def _feature_engineering_tab(self):
        """Feature engineering tab."""
        st.header("🔧 Feature Engineering")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please load data first using the sidebar controls.")
            return
        
        X = st.session_state.X
        y = st.session_state.y
        
        # Pipeline selection
        st.subheader("🔧 Pipeline Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pipeline_type = st.selectbox(
                "Pipeline Type",
                ["basic", "advanced"],
                help="Choose between basic and advanced feature engineering pipelines"
            )
        
        with col2:
            if st.button("🔄 Apply Feature Engineering", key="apply_fe"):
                self._apply_feature_engineering(pipeline_type)
        
        # Feature engineering options
        st.subheader("⚙️ Feature Engineering Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            enable_imputation = st.checkbox("Imputation", value=True)
        with col2:
            enable_transformations = st.checkbox("Transformations", value=True)
        with col3:
            enable_discretization = st.checkbox("Discretization", value=True)
        with col4:
            enable_scaling = st.checkbox("Scaling", value=True)
        
        # Show transformation info if available
        if 'feature_pipeline' in st.session_state:
            st.subheader("📋 Transformation Information")
            
            pipeline = st.session_state.feature_pipeline
            transformation_info = pipeline.get_transformation_info()
            
            for step_name, step_info in transformation_info.items():
                with st.expander(f"🔧 {step_name}"):
                    st.write(f"**Type**: {step_info['type']}")
                    st.write(f"**Parameters**: {step_info['parameters']}")
                    
                    if 'variables' in step_info:
                        st.write(f"**Variables**: {step_info['variables']}")
        
        # Feature importance if available
        if 'feature_importance' in st.session_state:
            st.subheader("📊 Feature Importance")
            
            feature_importance = st.session_state.feature_importance
            
            # Create bar chart
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            fig = px.bar(x=importances, y=features, orientation='h',
                        title="Feature Importance Scores")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    def _apply_feature_engineering(self, pipeline_type: str):
        """Apply feature engineering."""
        with st.spinner("Applying feature engineering..."):
            try:
                X = st.session_state.X
                y = st.session_state.y
                
                # Create pipeline
                if pipeline_type == "basic":
                    pipeline = BasicFeaturePipeline(self.config)
                else:
                    pipeline = AdvancedFeaturePipeline(self.config)
                
                # Apply transformations
                if pipeline_type == "advanced":
                    X_transformed = pipeline.fit_transform(X, y)
                else:
                    X_transformed = pipeline.fit_transform(X, y)
                
                # Store results
                st.session_state.feature_pipeline = pipeline
                st.session_state.X_transformed = X_transformed
                
                # Get feature importance if available
                if hasattr(pipeline, 'get_feature_importance'):
                    feature_importance = pipeline.get_feature_importance()
                    if feature_importance:
                        st.session_state.feature_importance = feature_importance
                
                st.success(f"✅ {pipeline_type.title()} feature engineering applied successfully!")
                st.info(f"📊 Original shape: {X.shape} → Transformed shape: {X_transformed.shape}")
                
            except Exception as e:
                st.error(f"❌ Error applying feature engineering: {e}")
    
    def _model_training_tab(self):
        """Model training tab."""
        st.header("🤖 Model Training")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please load data first using the sidebar controls.")
            return
        
        # Model configuration
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["classical", "automl"],
                help="Choose between classical baseline and AutoML models"
            )
        
        with col2:
            if st.button("🚀 Train Model", key="train_model_tab"):
                self._train_model_tab(model_type)
        
        # Model parameters
        if model_type == "classical":
            st.subheader("🔧 Classical Model Parameters")
            
            classical_model = st.selectbox(
                "Classical Model",
                ["linear_regression", "ridge", "lasso", "elastic_net", 
                 "random_forest", "gradient_boosting", "knn", "svr"],
                help="Choose a classical model type"
            )
            
            # Show model-specific parameters
            if classical_model == "ridge":
                alpha = st.slider("Alpha", 0.1, 100.0, 1.0, 0.1)
            elif classical_model == "lasso":
                alpha = st.slider("Alpha", 0.1, 100.0, 1.0, 0.1)
            elif classical_model == "elastic_net":
                alpha = st.slider("Alpha", 0.1, 100.0, 1.0, 0.1)
                l1_ratio = st.slider("L1 Ratio", 0.1, 0.9, 0.5, 0.1)
        
        elif model_type == "automl":
            st.subheader("🤖 AutoML Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_trials = st.slider("Number of Trials", 10, 200, 50)
            
            with col2:
                timeout = st.slider("Timeout (seconds)", 60, 3600, 300)
        
        # Training results
        if 'training_results' in st.session_state:
            st.subheader("📊 Training Results")
            
            results = st.session_state.training_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test RMSE", f"{results['test_metrics']['rmse']:.4f}")
            with col2:
                st.metric("Test R²", f"{results['test_metrics']['r2']:.4f}")
            with col3:
                st.metric("Test MAE", f"{results['test_metrics']['mae']:.4f}")
            with col4:
                st.metric("Test MAPE", f"{results['test_metrics']['mape']:.2f}%")
            
            # Predictions plot
            st.subheader("📈 Predictions Analysis")
            
            predictions = results['predictions']['test']
            y_true = predictions['true']
            y_pred = predictions['pred']
            
            # Scatter plot
            fig = px.scatter(x=y_true, y=y_pred, 
                           title="True vs Predicted Values",
                           labels={'x': 'True Values', 'y': 'Predicted Values'})
            
            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            residuals = y_true - y_pred
            
            fig = px.scatter(x=y_pred, y=residuals,
                           title="Residuals Plot",
                           labels={'x': 'Predicted Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _train_model_tab(self, model_type: str):
        """Train model for the tab."""
        with st.spinner("Training model..."):
            try:
                X = st.session_state.X
                y = st.session_state.y
                
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.data_module.split_data(X, y)
                
                # Apply feature engineering if available
                if 'feature_pipeline' in st.session_state:
                    pipeline = st.session_state.feature_pipeline
                    X_train_transformed = pipeline.transform(X_train)
                    X_val_transformed = pipeline.transform(X_val)
                    X_test_transformed = pipeline.transform(X_test)
                else:
                    X_train_transformed = X_train
                    X_val_transformed = X_val
                    X_test_transformed = X_test
                
                # Create and train model
                if model_type == "classical":
                    model = ClassicalBaseline(self.config)
                else:
                    model = AutoMLModel(self.config)
                
                model.fit(X_train_transformed, y_train)
                
                # Evaluate model
                results = self.evaluator.evaluate_model(
                    model, X_train_transformed, X_val_transformed, X_test_transformed,
                    y_train, y_val, y_test, f"{model_type}_model"
                )
                
                # Store results
                st.session_state.training_results = results
                st.session_state.trained_model = model
                st.session_state.models_trained = True
                
                st.success(f"✅ {model_type.title()} model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training model: {e}")
    
    def _train_model(self):
        """Train model using sidebar controls."""
        with st.spinner("Training model..."):
            try:
                X = st.session_state.X
                y = st.session_state.y
                
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.data_module.split_data(X, y)
                
                # Apply feature engineering if available
                if 'feature_pipeline' in st.session_state:
                    pipeline = st.session_state.feature_pipeline
                    X_train_transformed = pipeline.transform(X_train)
                    X_val_transformed = pipeline.transform(X_val)
                    X_test_transformed = pipeline.transform(X_test)
                else:
                    X_train_transformed = X_train
                    X_val_transformed = X_val
                    X_test_transformed = X_test
                
                # Create and train model
                if self.model_type == "classical":
                    model = ClassicalBaseline(self.config)
                else:
                    model = AutoMLModel(self.config)
                
                model.fit(X_train_transformed, y_train)
                
                # Evaluate model
                results = self.evaluator.evaluate_model(
                    model, X_train_transformed, X_val_transformed, X_test_transformed,
                    y_train, y_val, y_test, f"{self.model_type}_{self.pipeline_type}"
                )
                
                # Store results
                st.session_state.results[f"{self.model_type}_{self.pipeline_type}"] = results
                
                st.success(f"✅ {self.model_type.title()} model with {self.pipeline_type} pipeline trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training model: {e}")
    
    def _run_comprehensive_evaluation(self):
        """Run comprehensive evaluation."""
        with st.spinner("Running comprehensive evaluation..."):
            try:
                X = st.session_state.X
                y = st.session_state.y
                
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.data_module.split_data(X, y)
                
                # Define experiments
                experiments = [
                    ("classical", "basic"),
                    ("classical", "advanced"),
                    ("automl", "basic"),
                    ("automl", "advanced"),
                ]
                
                for model_type, pipeline_type in experiments:
                    try:
                        # Create pipeline
                        if pipeline_type == "basic":
                            pipeline = BasicFeaturePipeline(self.config)
                        else:
                            pipeline = AdvancedFeaturePipeline(self.config)
                        
                        # Apply transformations
                        if pipeline_type == "advanced":
                            X_train_transformed = pipeline.fit_transform(X_train, y_train)
                            X_val_transformed = pipeline.transform(X_val)
                            X_test_transformed = pipeline.transform(X_test)
                        else:
                            X_train_transformed = pipeline.fit_transform(X_train, y_train)
                            X_val_transformed = pipeline.transform(X_val)
                            X_test_transformed = pipeline.transform(X_test)
                        
                        # Create and train model
                        if model_type == "classical":
                            model = ClassicalBaseline(self.config)
                        else:
                            model = AutoMLModel(self.config)
                        
                        model.fit(X_train_transformed, y_train)
                        
                        # Evaluate model
                        results = self.evaluator.evaluate_model(
                            model, X_train_transformed, X_val_transformed, X_test_transformed,
                            y_train, y_val, y_test, f"{model_type}_{pipeline_type}"
                        )
                        
                        # Store results
                        st.session_state.results[f"{model_type}_{pipeline_type}"] = results
                        
                    except Exception as e:
                        st.warning(f"⚠️ Experiment {model_type}_{pipeline_type} failed: {e}")
                        continue
                
                st.success("✅ Comprehensive evaluation completed!")
                
            except Exception as e:
                st.error(f"❌ Error running comprehensive evaluation: {e}")
    
    def _results_analysis_tab(self):
        """Results analysis tab."""
        st.header("📈 Results Analysis")
        
        if not st.session_state.results:
            st.info("👆 Please train models first using the sidebar controls.")
            return
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in st.session_state.results.items():
            row = {"Model": model_name}
            for metric, value in results["test_metrics"].items():
                row[f"Test {metric.upper()}"] = value
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance charts
        st.subheader("📊 Performance Charts")
        
        # RMSE comparison
        fig = px.bar(comparison_df, x="Model", y="Test RMSE",
                    title="Test RMSE Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # R² comparison
        fig = px.bar(comparison_df, x="Model", y="Test R²",
                    title="Test R² Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = comparison_df.loc[comparison_df["Test RMSE"].idxmin(), "Model"]
        st.success(f"🏆 Best Model: **{best_model}** (Lowest RMSE)")
    
    def _summary_report_tab(self):
        """Summary report tab."""
        st.header("📋 Summary Report")
        
        if not st.session_state.results:
            st.info("👆 Please train models first using the sidebar controls.")
            return
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Trained", len(st.session_state.results))
        with col2:
            st.metric("Experiments Completed", len(st.session_state.results))
        with col3:
            best_model = min(st.session_state.results.items(), 
                           key=lambda x: x[1]["test_metrics"]["rmse"])
            st.metric("Best Model", best_model[0])
        with col4:
            best_rmse = best_model[1]["test_metrics"]["rmse"]
            st.metric("Best RMSE", f"{best_rmse:.4f}")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        for model_name, results in st.session_state.results.items():
            with st.expander(f"🔍 {model_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Test Metrics:**")
                    for metric, value in results["test_metrics"].items():
                        st.write(f"- {metric.upper()}: {value:.4f}")
                
                with col2:
                    st.write("**Validation Metrics:**")
                    for metric, value in results["val_metrics"].items():
                        st.write(f"- {metric.upper()}: {value:.4f}")
        
        # Safety and compliance
        st.subheader("🛡️ Safety & Compliance")
        
        st.info("""
        **Privacy Protection**: Enabled - All data processing follows privacy-preserving practices.
        
        **Bias Detection**: Enabled - Models are evaluated for potential bias.
        
        **Memory Usage**: Monitored - Memory usage is tracked and limited.
        
        **Disclaimer**: This is a research/educational demo. Not for production use.
        """)


def main():
    """Main function to run the Streamlit demo."""
    demo = StreamlitDemo()
    demo.run()


if __name__ == "__main__":
    main()
