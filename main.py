"""
Main script for Heart Disease Prediction Project

This script orchestrates the complete ML pipeline:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Model saving
"""

import argparse
import logging
from pathlib import Path

from data_preprocessing import DataPreprocessor
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']),
    format=config.LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)


def train_pipeline(model_name: str = None, tune_hyperparameters: bool = False):
    """
    Complete training pipeline
    
    Args:
        model_name: Specific model to train (None for all models)
        tune_hyperparameters: Whether to perform hyperparameter tuning
    """
    logger.info("="*70)
    logger.info("HEART DISEASE PREDICTION - TRAINING PIPELINE")
    logger.info("="*70)
    
    # Step 1: Data Preprocessing
    logger.info("\n[Step 1/4] Data Preprocessing...")
    preprocessor = DataPreprocessor(
        data_path=config.DATA_CONFIG['data_path'],
        test_size=config.DATA_CONFIG['test_size'],
        random_state=config.DATA_CONFIG['random_state']
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Step 2: Model Training
    logger.info("\n[Step 2/4] Model Training...")
    trainer = ModelTrainer(model_dir=config.MODEL_CONFIG['models_dir'])
    
    if model_name:
        # Train specific model
        if tune_hyperparameters and model_name in config.GRID_SEARCH_PARAMS:
            logger.info(f"Performing hyperparameter tuning for {model_name}...")
            trainer.hyperparameter_tuning(
                model_name, X_train, y_train,
                config.GRID_SEARCH_PARAMS[model_name]
            )
        else:
            hyperparams = config.HYPERPARAMETERS.get(model_name, {})
            trainer.train_model(model_name, X_train, y_train, hyperparams)
    else:
        # Train all models
        trainer.train_all_models(X_train, y_train)
    
    # Step 3: Model Evaluation
    logger.info("\n[Step 3/4] Model Evaluation...")
    evaluator = ModelEvaluator(output_dir=config.EVALUATION_CONFIG['output_dir'])
    
    if model_name:
        # Evaluate specific model
        model = trainer.models[model_name]
        evaluator.comprehensive_evaluation(
            model, X_test, y_test,
            model_name=model_name,
            feature_names=config.FEATURE_NAMES
        )
    else:
        # Compare all models
        comparison_df = evaluator.compare_models(trainer.models, X_test, y_test)
        
        # Select and evaluate best model
        best_model, best_model_name = trainer.select_best_model(X_test, y_test)
        evaluator.comprehensive_evaluation(
            best_model, X_test, y_test,
            model_name=best_model_name,
            feature_names=config.FEATURE_NAMES
        )
    
    # Step 4: Save Best Model
    logger.info("\n[Step 4/4] Saving Model...")
    if model_name:
        model_path = trainer.save_model(
            trainer.models[model_name],
            model_name,
            version='latest'
        )
    else:
        model_path = trainer.save_best_model(version='latest')
    
    logger.info(f"\nModel saved to: {model_path}")
    logger.info("\n" + "="*70)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Heart Disease Prediction - Training Pipeline'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic_regression', 'random_forest', 'naive_bayes', 'knn', 'decision_tree'],
        help='Specific model to train (default: train all models)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    args = parser.parse_args()
    
    try:
        train_pipeline(
            model_name=args.model,
            tune_hyperparameters=args.tune
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

