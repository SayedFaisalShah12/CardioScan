"""
Model Training Module for Heart Disease Prediction

This module handles:
- Model training with multiple algorithms
- Hyperparameter tuning
- Model persistence (saving/loading)
- Model comparison
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import logging
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training and managing ML models"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the ModelTrainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize available models
        self.available_models = {
            'logistic_regression': LogisticRegression(random_state=0, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=0),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=8),
            'decision_tree': DecisionTreeClassifier(max_depth=3, random_state=0)
        }
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                   hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            hyperparameters: Optional hyperparameters for the model
            
        Returns:
            Trained model
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.available_models.keys())}")
        
        logger.info(f"Training {model_name}...")
        
        # Get base model
        model = self.available_models[model_name]
        
        # Update hyperparameters if provided
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the model
        self.models[model_name] = model
        
        logger.info(f"{model_name} trained successfully")
        
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all available models...")
        
        for model_name in self.available_models.keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.models)} models")
        return self.models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with model performance metrics
        """
        if not self.models:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        results = []
        
        for model_name, model in self.models.items():
            train_score = model.score(X_test, y_test)  # Using test set for evaluation
            results.append({
                'model': model_name,
                'accuracy': train_score
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        logger.info("\nModel Performance:")
        logger.info(f"\n{results_df.to_string(index=False)}")
        
        return results_df
    
    def select_best_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, str]:
        """
        Select the best performing model
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Tuple of (best_model, best_model_name)
        """
        results_df = self.evaluate_models(X_test, y_test)
        
        self.best_model_name = results_df.iloc[0]['model']
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"\nBest model: {self.best_model_name} with accuracy: {results_df.iloc[0]['accuracy']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                             param_grid: Dict[str, list], cv: int = 5) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Dictionary of hyperparameters to search
            cv: Number of cross-validation folds
            
        Returns:
            Best model with tuned hyperparameters
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available")
        
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        base_model = self.available_models[model_name]
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Store the best model
        self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def save_model(self, model: Any, model_name: str, version: Optional[str] = None) -> str:
        """
        Save a trained model to disk
        
        Args:
            model: Trained model to save
            model_name: Name for the model file
            version: Optional version identifier (defaults to timestamp)
            
        Returns:
            Path to saved model file
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{model_name}_{version}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def save_best_model(self, version: Optional[str] = None) -> str:
        """
        Save the best model to disk
        
        Args:
            version: Optional version identifier
            
        Returns:
            Path to saved model file
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model() first.")
        
        return self.save_model(self.best_model, self.best_model_name, version)
    
    def cross_validate(self, model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model
        
        Args:
            model_name: Name of the model
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation metrics
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available")
        
        model = self.available_models[model_name]
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'min_accuracy': scores.min(),
            'max_accuracy': scores.max()
        }
        
        logger.info(f"\nCross-validation results for {model_name}:")
        logger.info(f"Mean Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']*2:.4f})")
        
        return results


def main():
    """Example usage of ModelTrainer"""
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(data_path='data/heart.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)
    
    # Evaluate and select best model
    trainer.select_best_model(X_test, y_test)
    
    # Save best model
    trainer.save_best_model()


if __name__ == "__main__":
    main()

