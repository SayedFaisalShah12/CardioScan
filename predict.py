"""
Prediction Module for Heart Disease Prediction

This module handles:
- Loading trained models
- Making predictions on new data
- Batch prediction
- Single instance prediction
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Dict, Any, Optional, Union, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """Class for making predictions using trained models"""
    
    def __init__(self, model_path: str):
        """
        Initialize the HeartDiseasePredictor
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the trained model from disk
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Make predictions on input data
        
        Args:
            X: Input features (DataFrame, array, or dictionary)
            
        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            # Assume it's a 2D array
            X = pd.DataFrame(X)
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame, array, or dictionary")
        
        try:
            predictions = self.model.predict(X)
            logger.info(f"Made predictions for {len(predictions)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features (DataFrame, array, or dictionary)
            
        Returns:
            Array of prediction probabilities [prob_class_0, prob_class_1]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        try:
            probabilities = self.model.predict_proba(X)
            logger.info(f"Got prediction probabilities for {len(probabilities)} samples")
            return probabilities
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {str(e)}")
            raise
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> pd.DataFrame:
        """
        Make predictions with confidence scores
        
        Args:
            X: Input features (DataFrame, array, or dictionary)
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        predictions = self.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.predict_proba(X)
            confidence = np.max(probabilities, axis=1)
            prob_disease = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        else:
            confidence = np.ones(len(predictions))
            prob_disease = predictions.astype(float)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'has_disease': predictions == 1,
            'probability_disease': prob_disease,
            'confidence': confidence
        })
        
        return results
    
    def predict_single(self, **kwargs) -> Dict[str, Any]:
        """
        Make prediction for a single instance using keyword arguments
        
        Args:
            **kwargs: Feature values as keyword arguments
            
        Returns:
            Dictionary with prediction results
        """
        # Create DataFrame from kwargs
        X = pd.DataFrame([kwargs])
        
        prediction = self.predict(X)[0]
        result = {
            'prediction': int(prediction),
            'has_disease': bool(prediction == 1),
            'prediction_label': 'Disease' if prediction == 1 else 'No Disease'
        }
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.predict_proba(X)[0]
            result['probability_no_disease'] = float(proba[0])
            result['probability_disease'] = float(proba[1])
            result['confidence'] = float(max(proba))
        
        return result
    
    def predict_batch(self, filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on a batch of data from a CSV file
        
        Args:
            filepath: Path to CSV file with input features
            output_path: Optional path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(data)} samples from {filepath}")
            
            predictions = self.predict_with_confidence(data)
            
            # Combine with original data
            results = pd.concat([data, predictions], axis=1)
            
            if output_path:
                results.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if the model supports it
        
        Returns:
            DataFrame with feature importance or None
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return None
        
        # Try to get feature names from the model or use indices
        try:
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                n_features = len(self.model.feature_importances_)
                feature_names = [f'feature_{i}' for i in range(n_features)]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None


def main():
    """Example usage of HeartDiseasePredictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Heart Disease Prediction')
    parser.add_argument('--model', type=str, default='models/best_model.pkl',
                       help='Path to trained model file')
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HeartDiseasePredictor(args.model)
    
    if args.input:
        # Batch prediction
        results = predictor.predict_batch(args.input, args.output)
        print(f"\nPredictions completed for {len(results)} samples")
        print(results.head())
    else:
        # Example single prediction
        example_data = {
            'age': 63,
            'sex': 1,
            'chest_pain_type': 3,
            'resting_blood_pressure': 145,
            'cholesterol': 233,
            'fasting_blood_sugar': 1,
            'rest_ecg': 0,
            'max_heart_rate_achieved': 150,
            'exercise_induced_angina': 0,
            'st_depression': 2.3,
            'st_slope': 0,
            'num_major_vessels': 0,
            'thalassemia': 1
        }
        
        result = predictor.predict_single(**example_data)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

