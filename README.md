# Automated Feature Engineering

A comprehensive research and educational project demonstrating automated feature engineering capabilities using classical and advanced AutoML approaches. This project showcases Meta/Hybrid AI techniques for automated feature discovery and model optimization.

## ⚠️ DISCLAIMER

**This is a research/educational demonstration. NOT for production decisions or control systems.**

- Results may vary and should be validated with domain experts
- Models are trained on limited datasets and may not generalize
- No warranty or guarantee of performance in real-world scenarios
- Use at your own risk for research and educational purposes only

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Automated-Feature-Engineering.git
cd Automated-Feature-Engineering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/streamlit_demo.py
```

### Basic Usage

```python
from src.train import AutomatedFeatureEngineeringTrainer

# Initialize trainer
trainer = AutomatedFeatureEngineeringTrainer("configs/config.yaml")

# Run comprehensive evaluation
results = trainer.run_comprehensive_evaluation()

# Print results
print(f"Best Model: {results['best_model']}")
print(results['comparison'])
```

## Dataset Schema

This project uses the California Housing dataset as a modern replacement for the deprecated Boston Housing dataset.

### Features

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| MedInc | Median income in block group | float | 0.5 - 15.0 |
| HouseAge | Median house age in block group | float | 1 - 52 |
| AveRooms | Average number of rooms per household | float | 0.8 - 141.9 |
| AveBedrms | Average number of bedrooms per household | float | 0.3 - 55.0 |
| Population | Block group population | float | 3 - 35682 |
| AveOccup | Average number of household members | float | 0.7 - 1243.3 |
| Latitude | Block group latitude | float | 32.5 - 41.9 |
| Longitude | Block group longitude | float | -124.3 - -114.3 |

### Target

| Target | Description | Type | Range |
|--------|-------------|------|-------|
| MedHouseVal | Median house value in block group | float | 0.1 - 5.0 |

### Data Splits

- **Training**: 60% of data
- **Validation**: 20% of data  
- **Test**: 20% of data

## Feature Engineering Pipelines

### Basic Pipeline

The basic pipeline includes standard feature engineering techniques:

1. **Missing Value Imputation**: Mean/median imputation for numeric features
2. **Log Transformation**: Applied to right-skewed features
3. **Power Transformation**: Yeo-Johnson transformation for normalization
4. **Discretization**: Equal-frequency binning for continuous features
5. **Feature Scaling**: Standard scaling for all features
6. **Feature Selection**: Removal of constant and duplicate features

### Advanced Pipeline

The advanced pipeline includes automated feature generation:

1. **Advanced Imputation**: Median imputation with outlier handling
2. **Yeo-Johnson Transformation**: Robust normalization
3. **Polynomial Features**: Interaction terms (degree=2, interaction_only=True)
4. **Mutual Information Selection**: Automated feature selection based on MI scores
5. **Robust Scaling**: Scaling using median and IQR

## Model Implementations

### Classical Baselines

- **Linear Regression**: Standard OLS regression
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization  
- **Elastic Net**: Combined L1/L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Gradient boosting regression
- **k-Nearest Neighbors**: Non-parametric regression
- **Support Vector Regression**: Kernel-based regression
- **Decision Tree**: Single decision tree

### Advanced AutoML

- **Optuna Optimization**: Bayesian optimization for hyperparameter tuning
- **Multi-Model Search**: Automatic model selection from multiple algorithms
- **Cross-Validation**: Robust evaluation with k-fold CV
- **Early Stopping**: Efficient optimization with timeout controls

## Evaluation Metrics

### Regression Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **Median AE**: Median Absolute Error
- **Max Error**: Maximum absolute error
- **Explained Variance**: Proportion of variance explained
- **MASE**: Mean Absolute Scaled Error

### Model Comparison

The evaluation framework provides comprehensive model comparison including:
- Cross-validation scores with confidence intervals
- Feature importance analysis
- Prediction analysis with residual plots
- Performance leaderboard

## Expected Performance Ranges

Based on the California Housing dataset:

| Model | Pipeline | Expected RMSE | Expected R² |
|-------|----------|---------------|-------------|
| Linear Regression | Basic | 0.65 - 0.75 | 0.55 - 0.65 |
| Random Forest | Basic | 0.50 - 0.60 | 0.70 - 0.80 |
| Gradient Boosting | Advanced | 0.45 - 0.55 | 0.75 - 0.85 |
| AutoML | Advanced | 0.40 - 0.50 | 0.80 - 0.90 |

*Note: Performance may vary based on random seeds and data splits*

## Training Commands

### Single Experiment

```bash
# Classical model with basic pipeline
python -m src.train --model classical --pipeline basic

# AutoML model with advanced pipeline  
python -m src.train --model automl --pipeline advanced
```

### Comprehensive Evaluation

```bash
# Run all model/pipeline combinations
python -m src.train --comprehensive
```

### Custom Configuration

```bash
# Use custom config file
python -m src.train --config configs/custom_config.yaml --comprehensive
```

## Interactive Demo

The Streamlit demo provides an interactive interface for exploring automated feature engineering:

### Features

- **Data Overview**: Dataset statistics, distributions, and correlations
- **Feature Engineering**: Interactive pipeline configuration and visualization
- **Model Training**: Real-time model training with progress tracking
- **Results Analysis**: Comprehensive performance comparison and visualization
- **Summary Reports**: Detailed analysis reports with safety compliance

### Launch Demo

```bash
streamlit run demo/streamlit_demo.py
```

Navigate to `http://localhost:8501` to access the interactive interface.

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_automated_feature_engineering.py -v
```

## Project Structure

```
automated-feature-engineering/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── boston_housing.py    # Dataset module
│   │   └── feature_engineering.py # Feature pipelines
│   ├── models/                   # Model implementations
│   │   ├── classical_baseline.py # Classical models
│   │   └── advanced_automl.py   # AutoML models
│   ├── metrics/                  # Evaluation metrics
│   ├── train/                    # Training scripts
│   ├── viz/                      # Visualization utilities
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── model/                    # Model configurations
│   ├── data/                     # Data configurations
│   └── feature_engineering/     # Pipeline configurations
├── demo/                         # Interactive demos
│   └── streamlit_demo.py        # Streamlit interface
├── tests/                        # Test suite
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   └── processed/               # Processed data
├── assets/                       # Generated assets
├── logs/                         # Log files
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## Safety & Compliance

### Privacy Protection

- **Data Anonymization**: All personal identifiers are removed
- **Privacy-Preserving**: No sensitive information is logged or stored
- **Consent**: Educational use only, no data collection

### Bias Detection

- **Fairness Metrics**: Model performance evaluated across different groups
- **Bias Monitoring**: Automated bias detection in predictions
- **Transparency**: Feature importance and model decisions are explainable

### Memory Management

- **Resource Limits**: Memory usage monitored and limited
- **Efficient Processing**: Optimized algorithms for large datasets
- **Cleanup**: Automatic cleanup of temporary files and memory

### Ethical Guidelines

- **Research Only**: Not intended for production decision-making
- **Transparency**: All methods and limitations clearly documented
- **Responsibility**: Users responsible for appropriate use
- **No Harm**: Designed to avoid harmful or discriminatory outcomes

## Research Applications

This project demonstrates several Meta/Hybrid AI concepts:

### Automated Feature Engineering

- **Feature Discovery**: Automatic identification of relevant features
- **Transformation Selection**: Optimal preprocessing pipeline selection
- **Feature Interaction**: Discovery of feature interactions and combinations

### Hyperparameter Optimization

- **Bayesian Optimization**: Efficient hyperparameter search using Optuna
- **Multi-Objective**: Optimization across multiple performance metrics
- **Early Stopping**: Resource-efficient optimization with timeout controls

### Model Selection

- **Automated ML**: Automatic model selection from multiple algorithms
- **Ensemble Methods**: Combination of multiple models for improved performance
- **Cross-Validation**: Robust model evaluation and selection

### Meta-Learning

- **Pipeline Optimization**: Learning optimal feature engineering pipelines
- **Transfer Learning**: Knowledge transfer across different datasets
- **Few-Shot Learning**: Efficient learning with limited data

## Educational Value

This project serves as an educational resource for:

- **Machine Learning Students**: Understanding automated feature engineering
- **Data Scientists**: Learning advanced preprocessing techniques
- **Researchers**: Exploring Meta/Hybrid AI approaches
- **Practitioners**: Understanding production-ready ML pipelines

### Learning Objectives

1. **Feature Engineering**: Understanding automated feature creation and selection
2. **Model Selection**: Learning automated model selection and optimization
3. **Evaluation**: Understanding comprehensive model evaluation frameworks
4. **Safety**: Learning about ethical AI and safety considerations
5. **Reproducibility**: Understanding reproducible research practices

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
ruff check src/ tests/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**kryptologyst**

- GitHub: [kryptologyst](https://github.com/kryptologyst)

## Acknowledgments

- **Feature-engine**: For providing comprehensive feature engineering tools
- **Optuna**: For advanced hyperparameter optimization capabilities
- **Scikit-learn**: For robust machine learning algorithms
- **Streamlit**: For interactive web application framework
- **California Housing Dataset**: For providing a modern, accessible dataset

## References

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

2. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining.

3. Soledad, S. (2021). Feature-engine: A Python library for feature engineering. Journal of Open Source Software, 6(57), 2709.

4. Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining.

5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.

## Future Work

- **Deep Learning Integration**: Neural network-based feature engineering
- **Time Series Support**: Specialized pipelines for temporal data
- **Categorical Features**: Advanced categorical feature engineering
- **Feature Selection**: More sophisticated feature selection algorithms
- **Distributed Computing**: Support for large-scale datasets
- **Model Interpretability**: Enhanced explainability features
- **Real-time Processing**: Streaming data support
- **Multi-modal Data**: Support for different data types
# Automated-Feature-Engineering
