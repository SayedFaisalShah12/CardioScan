"""
Data Preprocessing Module for Heart Disease Prediction

This module handles:
- Data loading
- Data cleaning and validation
- Feature engineering
- Train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing heart disease dataset"""
    
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 0):
        """
        Initialize the DataPreprocessor
        
        Args:
            data_path: Path to the CSV file
            test_size: Proportion of dataset to include in the test split
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def rename_columns(self) -> pd.DataFrame:
        """
        Rename columns to more descriptive names
        
        Returns:
            DataFrame with renamed columns
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before rename_columns()")
        
        column_mapping = {
            'age': 'age',
            'sex': 'sex',
            'cp': 'chest_pain_type',
            'trestbps': 'resting_blood_pressure',
            'chol': 'cholesterol',
            'fbs': 'fasting_blood_sugar',
            'restecg': 'rest_ecg',
            'thalach': 'max_heart_rate_achieved',
            'exang': 'exercise_induced_angina',
            'oldpeak': 'st_depression',
            'slope': 'st_slope',
            'ca': 'num_major_vessels',
            'thal': 'thalassemia',
            'target': 'target'
        }
        
        # Only rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in self.data.columns}
        self.data = self.data.rename(columns=existing_columns)
        logger.info("Columns renamed successfully")
        return self.data
    
    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset
        
        Returns:
            Series with count of missing values per column
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before check_missing_values()")
        
        missing_values = self.data.isnull().sum()
        total_missing = missing_values.sum()
        
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values in the dataset")
            logger.warning(f"Missing values per column:\n{missing_values[missing_values > 0]}")
        else:
            logger.info("No missing values found in the dataset")
        
        return missing_values
    
    def validate_data(self) -> bool:
        """
        Validate data quality and integrity
        
        Returns:
            True if data is valid, raises exception otherwise
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before validate_data()")
        
        # Check if target column exists
        if 'target' not in self.data.columns:
            raise ValueError("Target column 'target' not found in dataset")
        
        # Check for negative values in columns that shouldn't have them
        numeric_columns = ['age', 'resting_blood_pressure', 'cholesterol', 
                          'max_heart_rate_achieved', 'st_depression']
        
        for col in numeric_columns:
            if col in self.data.columns:
                if (self.data[col] < 0).any():
                    logger.warning(f"Found negative values in {col}")
        
        # Check target values (should be 0 or 1)
        if 'target' in self.data.columns:
            unique_targets = self.data['target'].unique()
            if not all(val in [0, 1] for val in unique_targets):
                logger.warning(f"Unexpected target values: {unique_targets}")
        
        logger.info("Data validation completed successfully")
        return True
    
    def get_feature_target_split(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target
        
        Returns:
            Tuple of (features, target)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before get_feature_target_split()")
        
        if 'target' not in self.data.columns:
            raise ValueError("Target column 'target' not found")
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before split_train_test()")
        
        X, y = self.get_feature_target_split()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        logger.info(f"Train target distribution:\n{self.y_train.value_counts()}")
        logger.info(f"Test target distribution:\n{self.y_test.value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        # Rename columns
        self.rename_columns()
        
        # Check for missing values
        self.check_missing_values()
        
        # Validate data
        self.validate_data()
        
        # Split into train/test
        self.split_train_test()
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_info(self) -> dict:
        """
        Get summary information about the dataset
        
        Returns:
            Dictionary with dataset information
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before get_data_info()")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'target_distribution': self.data['target'].value_counts().to_dict() if 'target' in self.data.columns else None
        }
        
        return info


def main():
    """Example usage of DataPreprocessor"""
    # Example usage
    preprocessor = DataPreprocessor(
        data_path='data/heart.csv',
        test_size=0.2,
        random_state=0
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    print("\nPreprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")


if __name__ == "__main__":
    main()

