# Early Dropout Risk Prediction System

# Project Overview
* The Early Dropout Risk Prediction System is a Machine Learning-based web application designed to identify students who are at risk of dropping out at an early stage.
* The system analyzes academic performance, demographic details, and enrollment-related features to predict whether a student is likely to continue or drop out. Early identification enables institutions to take preventive measures and provide necessary support to at-risk students.

 # Machine Learning Approach
 The project uses supervised machine learning techniques:

1. Algorithm Used:
  * Random Forest Classifier

2. Data Preprocessing:
   * Handling missing values
   * Feature scaling using StandardScaler
   * Categorical encoding using One-Hot Encoding

3. Model Evaluation Metrics:
   * Accuracy
   * Confusion Matrix
   * Classification Report

The trained model is saved and integrated into a Streamlit web application for real-time predictions.

# Libraries / Requirements.txt used
1. Pandas
2. Numpy
3. Scikit-learn
4. Joblib
5. Matplotlib
6. seaborn
7. os
8. Streamlit

# Streamlit Deployment
The model is deployed using Streamlit Cloud,enabling real-time student dropout risk prediction through a user-friendly web interface.
  # Deployed link
   https://early-dropout-prediction-yasha.streamlit.app

# Steps

1. Install requirements
   - pip install -r requirements.txt
2. Download the dataset
3. Data Preprocessing    
4. Run the application
   - streamlit run app.py
5. Required Files for Deployment
  * app.py
  * requirements.txt
  * model/random_forest_model.pkl
  * model/scaler.pkl
  * model/encoder.pkl
  * model/training_columns.pkl       

   

