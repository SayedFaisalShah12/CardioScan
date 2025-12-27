# CardioScan - Heart Disease Prediction

A production-level machine learning project for predicting heart disease using the UCI Heart Disease dataset.

## Project Structure

```
CardioScan/
├── data/                          # Data directory
│   └── heart.csv                  # Dataset file (place your data here)
├── models/                        # Trained models directory
├── evaluation_results/            # Evaluation plots and reports
├── output/                        # Prediction outputs
├── data_preprocessing.py          # Data loading and preprocessing module
├── train_model.py                 # Model training module
├── evaluate_model.py              # Model evaluation module
├── predict.py                     # Prediction module
├── config.py                      # Configuration file
├── main.py                        # Main training pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for preprocessing, training, evaluation, and prediction
- **Multiple Algorithms**: Support for Logistic Regression, Random Forest, Naive Bayes, KNN, and Decision Trees
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- **Model Persistence**: Save and load trained models for production use
- **Batch & Single Predictions**: Support for both batch processing and individual predictions
- **Hyperparameter Tuning**: Grid search for optimal hyperparameters
- **Production Ready**: Error handling, logging, and configuration management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CardioScan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset:
   - Download the UCI Heart Disease dataset
   - Place `heart.csv` in the `data/` directory

## Usage

### Training Models

Train all models:
```bash
python main.py
```

Train a specific model:
```bash
python main.py --model random_forest
```

Train with hyperparameter tuning:
```bash
python main.py --model random_forest --tune
```

### Comprehensive Evaluation

Run comprehensive evaluation with all metrics:
```bash
python evaluate_all_metrics.py
```

This will display:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-score
- ✅ Confusion Matrix
- ✅ ROC Curve and AUC
- ✅ Classification Report

### Streamlit Web Application

Launch the interactive web app:
```bash
streamlit run app.py
```

The Streamlit app provides:
- 🔮 **Single Patient Prediction**: Interactive form for individual predictions
- 📊 **Batch Prediction**: Upload CSV files for bulk predictions
- 📈 **Visualizations**: Probability distributions and feature importance
- 💾 **Download Results**: Export predictions as CSV

### Making Predictions

Single prediction using Python:
```python
from predict import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor('models/best_model_latest.pkl')

# Make prediction
result = predictor.predict_single(
    age=63,
    sex=1,
    chest_pain_type=3,
    resting_blood_pressure=145,
    cholesterol=233,
    fasting_blood_sugar=1,
    rest_ecg=0,
    max_heart_rate_achieved=150,
    exercise_induced_angina=0,
    st_depression=2.3,
    st_slope=0,
    num_major_vessels=0,
    thalassemia=1
)

print(result)
```

Batch prediction from CSV:
```bash
python predict.py --model models/best_model_latest.pkl --input data/test_data.csv --output output/predictions.csv
```

### Using Individual Modules

#### Data Preprocessing
```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(data_path='data/heart.csv')
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
```

#### Model Training
```python
from train_model import ModelTrainer

trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train)
best_model, best_name = trainer.select_best_model(X_test, y_test)
trainer.save_best_model()
```

#### Model Evaluation
```python
from evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.comprehensive_evaluation(
    model, X_test, y_test,
    model_name='Random Forest'
)
```

## Dataset Information

The UCI Heart Disease dataset contains the following features:

1. **age**: Age in years
2. **sex**: Sex (1 = male, 0 = female)
3. **cp**: Chest pain type (1-4)
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise
11. **slope**: Slope of peak exercise ST segment (1-3)
12. **ca**: Number of major vessels colored by flourosopy (0-3)
13. **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
14. **target**: Heart disease (0 = no, 1 = yes)

## Model Performance

The project supports multiple algorithms. Typical performance on the UCI dataset:

- **Random Forest**: ~88-90% accuracy
- **Logistic Regression**: ~85% accuracy
- **Naive Bayes**: ~85% accuracy
- **Decision Tree**: ~82% accuracy
- **KNN**: ~69% accuracy

*Note: Actual performance may vary based on data and hyperparameters*

## Configuration

Edit `config.py` to customize:
- Data paths and parameters
- Model hyperparameters
- Evaluation settings
- Logging configuration

## Output Files

After training, you'll find:
- **models/**: Saved model files (.pkl)
- **evaluation_results/**: 
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Feature importance plots

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- joblib >= 1.2.0
- streamlit >= 1.28.0
- plotly >= 5.17.0

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.
