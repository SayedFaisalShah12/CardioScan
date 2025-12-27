"""
Comprehensive Evaluation Script

This script evaluates a trained model and displays all metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

from data_preprocessing import DataPreprocessor
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def display_evaluation_metrics(model, X_test, y_test, model_name="Model"):
    """
    Display comprehensive evaluation metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
    """
    print("\n" + "="*70)
    print(f"COMPREHENSIVE EVALUATION: {model_name.upper()}")
    print("="*70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            pass
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Display metrics
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Confusion Matrix
    print("\n📋 CONFUSION MATRIX")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n                Predicted")
    print(f"              No Disease  Disease")
    print(f"Actual No Disease    {tn:4d}      {fp:4d}")
    print(f"       Disease       {fn:4d}      {tp:4d}")
    
    print(f"\nTrue Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Disease', 'Disease'],
               yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'evaluation_results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    print(f"\n✓ Confusion matrix saved to evaluation_results/")
    
    # ROC Curve
    if y_pred_proba is not None:
        print("\n📈 ROC CURVE")
        print("-" * 70)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'evaluation_results/roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        print(f"AUC Score: {auc_score:.4f}")
        print(f"✓ ROC curve saved to evaluation_results/")
    
    # Classification Report
    print("\n📄 CLASSIFICATION REPORT")
    print("-" * 70)
    report = classification_report(y_test, y_pred,
                                  target_names=['No Disease', 'Disease'])
    print(report)
    
    print("\n" + "="*70 + "\n")


def main():
    """Main evaluation function"""
    print("="*70)
    print("HEART DISEASE PREDICTION - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Load and preprocess data
    print("\n[Step 1/3] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(
        data_path=config.DATA_CONFIG['data_path'],
        test_size=config.DATA_CONFIG['test_size'],
        random_state=config.DATA_CONFIG['random_state']
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Train models
    print("\n[Step 2/3] Training models...")
    trainer = ModelTrainer(model_dir=config.MODEL_CONFIG['models_dir'])
    trainer.train_all_models(X_train, y_train)
    
    # Evaluate all models
    print("\n[Step 3/3] Evaluating models...")
    evaluator = ModelEvaluator(output_dir=config.EVALUATION_CONFIG['output_dir'])
    
    # Evaluate each model
    for model_name, model in trainer.models.items():
        display_evaluation_metrics(model, X_test, y_test, model_name)
    
    # Select and display best model
    print("\n🏆 SELECTING BEST MODEL")
    print("="*70)
    best_model, best_model_name = trainer.select_best_model(X_test, y_test)
    
    print(f"\nBest Model: {best_model_name}")
    print("\nFinal Evaluation of Best Model:")
    display_evaluation_metrics(best_model, X_test, y_test, best_model_name)
    
    # Save best model
    model_path = trainer.save_best_model(version='best_model', use_pickle=True)
    print(f"\n✓ Best model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure heart.csv is in the data/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

