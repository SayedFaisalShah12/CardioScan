"""
Configuration Module for Heart Disease Prediction Project

This module contains all configuration parameters for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'evaluation_results'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'data_path': str(DATA_DIR / 'heart.csv'),
    'test_size': 0.2,
    'random_state': 0,
    'target_column': 'target'
}

# Model configuration
MODEL_CONFIG = {
    'models_dir': str(MODELS_DIR),
    'default_model': 'random_forest',
    'random_state': 0,
    'cv_folds': 5
}

# Model hyperparameters
HYPERPARAMETERS = {
    'logistic_regression': {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 0
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 0
    },
    'naive_bayes': {
        # GaussianNB has no hyperparameters to tune
    },
    'knn': {
        'n_neighbors': 8,
        'weights': 'uniform',
        'algorithm': 'auto'
    },
    'decision_tree': {
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 0
    }
}

# Grid search parameters for hyperparameter tuning
GRID_SEARCH_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10]
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'output_dir': str(RESULTS_DIR),
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300
}

# Feature names (after preprocessing)
FEATURE_NAMES = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_blood_pressure',
    'cholesterol',
    'fasting_blood_sugar',
    'rest_ecg',
    'max_heart_rate_achieved',
    'exercise_induced_angina',
    'st_depression',
    'st_slope',
    'num_major_vessels',
    'thalassemia'
]

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': None  # Set to a file path to enable file logging
}

# Prediction configuration
PREDICTION_CONFIG = {
    'default_model_path': str(MODELS_DIR / 'best_model.pkl'),
    'confidence_threshold': 0.5
}

