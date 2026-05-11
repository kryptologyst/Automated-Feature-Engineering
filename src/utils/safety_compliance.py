"""Safety and compliance module for automated feature engineering."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class SafetyCompliance:
    """Safety and compliance checker for automated feature engineering."""
    
    def __init__(self, config: DictConfig):
        """Initialize safety compliance checker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Safety thresholds
        self.max_memory_gb = config.safety.max_memory_usage_gb
        self.enable_privacy_protection = config.safety.enable_privacy_protection
        self.enable_bias_detection = config.safety.enable_bias_detection
        
        self.logger.info("Safety compliance checker initialized")
    
    def check_data_privacy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data for privacy violations.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with privacy check results
        """
        if not self.enable_privacy_protection:
            return {"status": "disabled", "violations": []}
        
        violations = []
        
        # Check for potential PII in column names
        pii_keywords = ["name", "email", "phone", "ssn", "id", "address", "zip"]
        for col in data.columns:
            if any(keyword in col.lower() for keyword in pii_keywords):
                violations.append(f"Potential PII in column name: {col}")
        
        # Check for potential PII in data values
        for col in data.columns:
            if data[col].dtype == "object":
                # Check for email patterns
                email_pattern = data[col].astype(str).str.contains(r'@', na=False)
                if email_pattern.any():
                    violations.append(f"Potential email addresses in column: {col}")
                
                # Check for phone patterns
                phone_pattern = data[col].astype(str).str.contains(r'\d{3}-\d{3}-\d{4}', na=False)
                if phone_pattern.any():
                    violations.append(f"Potential phone numbers in column: {col}")
        
        status = "passed" if not violations else "failed"
        
        return {
            "status": status,
            "violations": violations,
            "recommendations": self._get_privacy_recommendations(violations)
        }
    
    def check_model_bias(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Check model for potential bias.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with bias check results
        """
        if not self.enable_bias_detection:
            return {"status": "disabled", "bias_metrics": {}}
        
        bias_metrics = {}
        
        # Check for feature importance bias
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            
            # Check for extreme feature importance
            max_importance = np.max(feature_importance)
            min_importance = np.min(feature_importance)
            
            if max_importance > 0.8:
                bias_metrics["extreme_feature_importance"] = {
                    "status": "warning",
                    "message": "One feature dominates predictions (>80% importance)"
                }
            
            # Check for feature importance distribution
            importance_std = np.std(feature_importance)
            if importance_std > 0.3:
                bias_metrics["uneven_feature_importance"] = {
                    "status": "warning", 
                    "message": "Uneven distribution of feature importance"
                }
        
        # Check for prediction bias
        if hasattr(model, "predict"):
            predictions = model.predict(X)
            
            # Check for prediction range bias
            pred_range = np.max(predictions) - np.min(predictions)
            target_range = np.max(y) - np.min(y)
            
            if pred_range < 0.1 * target_range:
                bias_metrics["limited_prediction_range"] = {
                    "status": "warning",
                    "message": "Model predictions have limited range"
                }
        
        status = "passed" if not bias_metrics else "warning"
        
        return {
            "status": status,
            "bias_metrics": bias_metrics,
            "recommendations": self._get_bias_recommendations(bias_metrics)
        }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            import psutil
            
            # Get memory usage
            memory_info = psutil.virtual_memory()
            used_gb = memory_info.used / (1024**3)
            available_gb = memory_info.available / (1024**3)
            
            status = "passed" if used_gb < self.max_memory_gb else "warning"
            
            return {
                "status": status,
                "used_gb": used_gb,
                "available_gb": available_gb,
                "max_allowed_gb": self.max_memory_gb,
                "usage_percentage": (used_gb / self.max_memory_gb) * 100
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "psutil not available for memory monitoring"
            }
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with data quality results
        """
        quality_issues = []
        
        # Check for missing values
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        high_missing = missing_percentage[missing_percentage > 50]
        
        if not high_missing.empty:
            quality_issues.append({
                "type": "high_missing_values",
                "message": f"Columns with >50% missing values: {list(high_missing.index)}",
                "severity": "warning"
            })
        
        # Check for constant features
        constant_features = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            quality_issues.append({
                "type": "constant_features",
                "message": f"Constant features found: {constant_features}",
                "severity": "warning"
            })
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append({
                "type": "duplicate_rows",
                "message": f"Duplicate rows found: {duplicate_count}",
                "severity": "info"
            })
        
        # Check for outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_issues = []
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > len(data) * 0.1:  # More than 10% outliers
                outlier_issues.append(col)
        
        if outlier_issues:
            quality_issues.append({
                "type": "high_outliers",
                "message": f"Columns with high outlier percentage: {outlier_issues}",
                "severity": "info"
            })
        
        status = "passed" if not quality_issues else "warning"
        
        return {
            "status": status,
            "issues": quality_issues,
            "recommendations": self._get_quality_recommendations(quality_issues)
        }
    
    def run_comprehensive_safety_check(
        self, 
        data: pd.DataFrame, 
        model: Any = None, 
        target: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Run comprehensive safety checks.
        
        Args:
            data: DataFrame to check
            model: Trained model (optional)
            target: Target variable (optional)
            
        Returns:
            Dictionary with comprehensive safety check results
        """
        self.logger.info("Running comprehensive safety checks")
        
        results = {
            "privacy_check": self.check_data_privacy(data),
            "memory_check": self.check_memory_usage(),
            "data_quality_check": self.check_data_quality(data)
        }
        
        if model is not None and target is not None:
            results["bias_check"] = self.check_model_bias(model, data, target)
        
        # Overall status
        all_statuses = [check["status"] for check in results.values()]
        if "error" in all_statuses:
            overall_status = "error"
        elif "warning" in all_statuses:
            overall_status = "warning"
        else:
            overall_status = "passed"
        
        results["overall_status"] = overall_status
        
        # Generate summary
        results["summary"] = self._generate_safety_summary(results)
        
        self.logger.info(f"Safety checks completed with status: {overall_status}")
        
        return results
    
    def _get_privacy_recommendations(self, violations: List[str]) -> List[str]:
        """Get privacy protection recommendations.
        
        Args:
            violations: List of privacy violations
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if violations:
            recommendations.extend([
                "Remove or anonymize personally identifiable information",
                "Use data masking techniques for sensitive columns",
                "Implement data access controls and audit logging",
                "Consider differential privacy for sensitive datasets",
                "Review data retention policies"
            ])
        
        return recommendations
    
    def _get_bias_recommendations(self, bias_metrics: Dict[str, Any]) -> List[str]:
        """Get bias mitigation recommendations.
        
        Args:
            bias_metrics: Dictionary with bias metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if bias_metrics:
            recommendations.extend([
                "Review feature importance distribution",
                "Consider feature selection techniques to reduce bias",
                "Implement fairness constraints in model training",
                "Use diverse training data to reduce bias",
                "Monitor model performance across different groups",
                "Consider using bias-aware evaluation metrics"
            ])
        
        return recommendations
    
    def _get_quality_recommendations(self, quality_issues: List[Dict[str, Any]]) -> List[str]:
        """Get data quality improvement recommendations.
        
        Args:
            quality_issues: List of quality issues
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if quality_issues:
            recommendations.extend([
                "Address missing values using appropriate imputation strategies",
                "Remove or transform constant features",
                "Handle duplicate rows appropriately",
                "Consider outlier detection and treatment",
                "Implement data validation pipelines",
                "Use data quality monitoring tools"
            ])
        
        return recommendations
    
    def _generate_safety_summary(self, results: Dict[str, Any]) -> str:
        """Generate safety check summary.
        
        Args:
            results: Dictionary with safety check results
            
        Returns:
            Summary string
        """
        overall_status = results["overall_status"]
        
        if overall_status == "passed":
            return "All safety checks passed. The system is ready for use."
        elif overall_status == "warning":
            return "Some safety checks generated warnings. Review recommendations before proceeding."
        else:
            return "Safety checks failed. Address critical issues before proceeding."
    
    def generate_safety_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed safety report.
        
        Args:
            results: Dictionary with safety check results
            
        Returns:
            Detailed safety report
        """
        report = f"""
# Safety Compliance Report

## Overall Status: {results['overall_status'].upper()}

## Privacy Check
- Status: {results['privacy_check']['status']}
- Violations: {len(results['privacy_check']['violations'])}
- Recommendations: {len(results['privacy_check']['recommendations'])}

## Memory Usage Check  
- Status: {results['memory_check']['status']}
- Used: {results['memory_check'].get('used_gb', 'N/A'):.2f} GB
- Available: {results['memory_check'].get('available_gb', 'N/A'):.2f} GB

## Data Quality Check
- Status: {results['data_quality_check']['status']}
- Issues Found: {len(results['data_quality_check']['issues'])}

## Bias Check
- Status: {results.get('bias_check', {}).get('status', 'N/A')}
- Bias Metrics: {len(results.get('bias_check', {}).get('bias_metrics', {}))}

## Summary
{results['summary']}

## Recommendations
- Review all warnings and recommendations
- Implement appropriate safety measures
- Monitor system performance continuously
- Update safety policies as needed

---
Generated by Automated Feature Engineering Safety Compliance System
        """
        
        return report
