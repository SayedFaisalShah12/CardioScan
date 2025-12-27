"""
Streamlit Web Application for Heart Disease Prediction

This app provides an interactive interface for:
- Making heart disease predictions
- Viewing prediction results with confidence scores
- Understanding feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from predict import HeartDiseasePredictor
import config

# Page configuration
st.set_page_config(
    page_title="CardioScan - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model (cached for performance)"""
    try:
        predictor = HeartDiseasePredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def get_default_model_path():
    """Get the default model path"""
    # Try to find the latest model
    models_dir = Path(config.MODEL_CONFIG['models_dir'])
    if models_dir.exists():
        # Look for best_model files
        model_files = list(models_dir.glob('*best_model*.pkl'))
        if model_files:
            # Get the most recent one
            latest_model = max(model_files, key=os.path.getmtime)
            return str(latest_model)
    
    # Fall back to default
    return config.PREDICTION_CONFIG['default_model_path']


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">❤️ CardioScan</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value=get_default_model_path(),
        help="Path to the trained model file"
    )
    
    # Load model
    predictor = load_model(model_path)
    
    if predictor is None:
        st.error("⚠️ Could not load model. Please check the model path in the sidebar.")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Batch Prediction", "ℹ️ About"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Patient Information")
        st.markdown("Enter patient details to predict heart disease risk.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=63, step=1)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            
            st.subheader("Clinical Measurements")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=50, max_value=250, value=145, step=1
            )
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100, max_value=600, value=233, step=1
            )
            max_heart_rate = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=60, max_value=220, value=150, step=1
            )
            st_depression = st.number_input(
                "ST Depression (exercise relative to rest)",
                min_value=0.0, max_value=10.0, value=2.3, step=0.1, format="%.1f"
            )
        
        with col2:
            st.subheader("Symptoms & Tests")
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=[
                    ("Typical Angina", 1),
                    ("Atypical Angina", 2),
                    ("Non-anginal Pain", 3),
                    ("Asymptomatic", 4)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            fasting_blood_sugar = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0]
            )[1]
            
            rest_ecg = st.selectbox(
                "Resting Electrocardiographic Results",
                options=[
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            exercise_angina = st.selectbox(
                "Exercise Induced Angina",
                options=[("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0]
            )[1]
            
            st_slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[
                    ("Upsloping", 1),
                    ("Flat", 2),
                    ("Downsloping", 3)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            num_major_vessels = st.number_input(
                "Number of Major Vessels (0-3)",
                min_value=0, max_value=3, value=0, step=1
            )
            
            thalassemia = st.selectbox(
                "Thalassemia",
                options=[
                    ("Normal", 3),
                    ("Fixed Defect", 6),
                    ("Reversible Defect", 7)
                ],
                format_func=lambda x: x[0]
            )[1]
        
        # Prediction button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            predict_button = st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True)
        
        if predict_button:
            # Prepare input data
            input_data = {
                'age': age,
                'sex': sex,
                'chest_pain_type': chest_pain_type,
                'resting_blood_pressure': resting_bp,
                'cholesterol': cholesterol,
                'fasting_blood_sugar': fasting_blood_sugar,
                'rest_ecg': rest_ecg,
                'max_heart_rate_achieved': max_heart_rate,
                'exercise_induced_angina': exercise_angina,
                'st_depression': st_depression,
                'st_slope': st_slope,
                'num_major_vessels': num_major_vessels,
                'thalassemia': thalassemia
            }
            
            # Make prediction
            try:
                result = predictor.predict_single(**input_data)
                
                # Display results
                st.markdown("---")
                
                # Prediction result box
                if result['has_disease']:
                    st.markdown("""
                        <div class="prediction-box">
                            <h2>⚠️ HIGH RISK DETECTED</h2>
                            <p style="font-size: 1.2rem;">The model predicts: <strong>Heart Disease Present</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-box" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                            <h2>✅ LOW RISK</h2>
                            <p style="font-size: 1.2rem;">The model predicts: <strong>No Heart Disease</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                
                with col_met1:
                    st.metric(
                        "Prediction",
                        result['prediction_label'],
                        delta=None
                    )
                
                if 'probability_disease' in result:
                    with col_met2:
                        prob_disease = result['probability_disease']
                        st.metric(
                            "Disease Probability",
                            f"{prob_disease:.1%}",
                            delta=None
                        )
                    
                    with col_met3:
                        confidence = result.get('confidence', prob_disease)
                        st.metric(
                            "Confidence",
                            f"{confidence:.1%}",
                            delta=None
                        )
                    
                    # Probability visualization
                    st.markdown("### Probability Distribution")
                    prob_data = pd.DataFrame({
                        'Outcome': ['No Disease', 'Disease'],
                        'Probability': [
                            result.get('probability_no_disease', 1 - prob_disease),
                            prob_disease
                        ]
                    })
                    
                    fig = px.bar(
                        prob_data,
                        x='Outcome',
                        y='Probability',
                        color='Outcome',
                        color_discrete_map={'No Disease': '#38ef7d', 'Disease': '#e63946'},
                        text='Probability'
                    )
                    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig.update_layout(
                        yaxis_title="Probability",
                        xaxis_title="",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (if available)
                importance_df = predictor.get_feature_importance()
                if importance_df is not None:
                    st.markdown("### Feature Importance")
                    top_features = importance_df.head(10)
                    
                    fig_imp = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        color='importance',
                        color_continuous_scale='Reds',
                        labels={'importance': 'Importance', 'feature': 'Feature'}
                    )
                    fig_imp.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with patient data to make predictions for multiple patients.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should contain columns: age, sex, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate_achieved, exercise_induced_angina, st_depression, st_slope, num_major_vessels, thalassemia"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} records")
                
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Make predictions
                if st.button("🔍 Predict for All Patients", type="primary"):
                    try:
                        results = predictor.predict_with_confidence(df)
                        
                        st.success("✅ Predictions completed!")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        
                        with col_sum1:
                            total = len(results)
                            st.metric("Total Patients", total)
                        
                        with col_sum2:
                            with_disease = results['has_disease'].sum()
                            st.metric("Predicted with Disease", with_disease, delta=f"{with_disease/total:.1%}")
                        
                        with col_sum3:
                            avg_confidence = results['confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="heart_disease_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Tab 3: About
    with tab3:
        st.header("About CardioScan")
        st.markdown("""
        ### Overview
        CardioScan is a machine learning-based heart disease prediction system that uses 
        patient clinical data to assess the risk of heart disease.
        
        ### Features
        - **Single Patient Prediction**: Enter individual patient data for instant risk assessment
        - **Batch Processing**: Upload CSV files for bulk predictions
        - **Confidence Scores**: Get probability estimates for predictions
        - **Feature Importance**: Understand which factors contribute most to predictions
        
        ### Model Information
        The system uses a trained machine learning model (Random Forest, Logistic Regression, 
        or other algorithms) to make predictions based on:
        
        - Demographics (age, sex)
        - Clinical measurements (blood pressure, cholesterol, heart rate)
        - Symptoms and test results (chest pain, ECG results, exercise tests)
        
        ### Data Requirements
        The model requires the following 13 features:
        1. Age
        2. Sex
        3. Chest Pain Type
        4. Resting Blood Pressure
        5. Cholesterol
        6. Fasting Blood Sugar
        7. Resting ECG Results
        8. Maximum Heart Rate Achieved
        9. Exercise Induced Angina
        10. ST Depression
        11. ST Slope
        12. Number of Major Vessels
        13. Thalassemia
        
        ### Disclaimer
        ⚠️ **This tool is for educational and research purposes only.**
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
        Always consult with qualified healthcare providers for medical decisions.
        """)


if __name__ == "__main__":
    main()

