"""
Model Evaluation Module for Heart Disease Prediction

This module handles:
- Comprehensive model evaluation metrics
- Confusion matrix analysis
- Classification reports
- ROC curve and AUC
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating ML models"""
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        """
        Initialize the ModelEvaluator
        
        Args:
            output_dir: Directory to save evaluation plots and reports
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print evaluation metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics to print
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")
        print("="*50 + "\n")
    
    def confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = "Model", save_plot: bool = True) -> np.ndarray:
        """
        Generate and visualize confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model (for plot title)
            save_plot: Whether to save the plot
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_plot:
            filepath = os.path.join(self.output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {filepath}")
        
        plt.close()
        
        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix for {model_name}:")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")
        
        return cm
    
    def classification_report_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      target_names: Optional[list] = None) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Optional names for target classes
            
        Returns:
            Classification report as string
        """
        if target_names is None:
            target_names = ['No Disease', 'Disease']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\nClassification Report:")
        print(report)
        
        return report
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model", save_plot: bool = True) -> Tuple[float, float]:
        """
        Plot ROC curve and calculate AUC
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (fpr, tpr) arrays
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            filepath = os.path.join(self.output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {filepath}")
        
        plt.close()
        
        return fpr, tpr
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    model_name: str = "Model", save_plot: bool = True) -> Tuple[float, float]:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (precision, recall) arrays
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            filepath = os.path.join(self.output_dir, f'pr_curve_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {filepath}")
        
        plt.close()
        
        return precision, recall
    
    def feature_importance_analysis(self, model: Any, feature_names: list,
                                   model_name: str = "Model", top_n: int = 10,
                                   save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze and visualize feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
            save_plot: Whether to save the plot
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_plot:
            filepath = os.path.join(self.output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {filepath}")
        
        plt.close()
        
        return importance_df
    
    def comprehensive_evaluation(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                model_name: str = "Model", feature_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"Performing comprehensive evaluation for {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            except:
                logger.warning("Could not get prediction probabilities")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        self.print_metrics(metrics)
        
        # Confusion matrix
        cm = self.confusion_matrix_analysis(y_test, y_pred, model_name)
        
        # Classification report
        report = self.classification_report_analysis(y_test, y_pred)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # ROC curve if probabilities available
        if y_pred_proba is not None:
            fpr, tpr = self.plot_roc_curve(y_test, y_pred_proba, model_name)
            precision, recall = self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
            results['roc_curve'] = (fpr, tpr)
            results['pr_curve'] = (precision, recall)
        
        # Feature importance if available
        if feature_names is None:
            feature_names = list(X_test.columns)
        
        if hasattr(model, 'feature_importances_'):
            importance_df = self.feature_importance_analysis(model, feature_names, model_name)
            results['feature_importance'] = importance_df
        
        logger.info(f"Comprehensive evaluation completed for {model_name}")
        
        return results
    
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models side by side
        
        Args:
            models: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            metrics['model'] = model_name
            comparison_results.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70 + "\n")
        
        return comparison_df


def main():
    """Example usage of ModelEvaluator"""
    import joblib
    from data_preprocessing import DataPreprocessor
    from train_model import ModelTrainer
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(data_path='data/heart.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Train a model
    trainer = ModelTrainer()
    trainer.train_model('random_forest', X_train, y_train)
    model = trainer.models['random_forest']
    
    # Evaluate model
    evaluator = ModelEvaluator()
    results = evaluator.comprehensive_evaluation(
        model, X_test, y_test,
        model_name='Random Forest',
        feature_names=list(X_test.columns)
    )


if __name__ == "__main__":
    main()

