"""
Script to train models and save the best-performing model using pickle

This script:
1. Loads and preprocesses the data
2. Trains all available models
3. Evaluates and selects the best model
4. Saves the best model using pickle
"""

import logging
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


def save_best_model_with_pickle():
    """
    Complete pipeline to train models and save the best one using pickle
    """
    logger.info("="*70)
    logger.info("SAVING BEST MODEL USING PICKLE")
    logger.info("="*70)
    
    # Step 1: Data Preprocessing
    logger.info("\n[Step 1/3] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(
        data_path=config.DATA_CONFIG['data_path'],
        test_size=config.DATA_CONFIG['test_size'],
        random_state=config.DATA_CONFIG['random_state']
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Step 2: Train all models
    logger.info("\n[Step 2/3] Training all models...")
    trainer = ModelTrainer(model_dir=config.MODEL_CONFIG['models_dir'])
    trainer.train_all_models(X_train, y_train)
    
    # Step 3: Evaluate and select best model
    logger.info("\n[Step 3/3] Evaluating models and selecting best...")
    evaluator = ModelEvaluator(output_dir=config.EVALUATION_CONFIG['output_dir'])
    
    # Compare all models
    comparison_df = evaluator.compare_models(trainer.models, X_test, y_test)
    
    # Select best model
    best_model, best_model_name = trainer.select_best_model(X_test, y_test)
    
    # Comprehensive evaluation of best model
    evaluator.comprehensive_evaluation(
        best_model, X_test, y_test,
        model_name=best_model_name,
        feature_names=config.FEATURE_NAMES
    )
    
    # Save best model using pickle
    logger.info(f"\nSaving best model '{best_model_name}' using pickle...")
    model_path = trainer.save_best_model(version='best_model', use_pickle=True)
    
    logger.info("\n" + "="*70)
    logger.info(f"SUCCESS! Best model saved using pickle to: {model_path}")
    logger.info("="*70)
    
    return model_path, best_model_name


if __name__ == "__main__":
    try:
        model_path, model_name = save_best_model_with_pickle()
        print(f"\n✓ Best model '{model_name}' saved successfully!")
        print(f"✓ Model file: {model_path}")
        print(f"\nYou can now use this model for predictions:")
        print(f"  from predict import HeartDiseasePredictor")
        print(f"  predictor = HeartDiseasePredictor('{model_path}')")
    except FileNotFoundError as e:
        logger.error(f"Data file not found. Please ensure heart.csv is in the data/ directory.")
        logger.error(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

