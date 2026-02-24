import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models
model=joblib.load("model/random_forest_model.pkl")
scaler=joblib.load("model/scaler.pkl")
encoder=joblib.load("model/encoder.pkl")
training_columns=joblib.load("model/training_columns.pkl")

# Title of the app
st.title("Bulk Dropout Prediction System")

# Upload the dataset
uploaded_file = st.file_uploader("Upload Student Dataset", type=["csv"]) 

if uploaded_file:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    # Original data for display
    original_data = data.copy()

    # Boolean mapping
    boolean_cols={
        "Daytime/evening attendance": {1: True, 0: False},
        "Displaced": {1: True, 0: False},
        "Educational special needs": {1: True, 0: False},
        "Debtor": {1: True, 0: False},
        "Tuition fees up to date": {1: True, 0: False},
        "Gender": {1: True, 0: False},
        "Scholarship holder": {1: True, 0: False},
        "International": {1: True, 0: False},
    }

    # Apply boolean mapping
    for col, mapping in boolean_cols.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
    # Categorical columns
    categorical_cols = [
        "Marital status", "Application mode", "Course",
        "Previous qualification","Nacionality",
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation"
    ]
    # Check missing columns
    missing_cols = set(categorical_cols) - set(data.columns)
    if missing_cols:
        st.error(f"Missing columns : {missing_cols}")
        st.stop()

    # Encode categorical variables
    encoded=encoder.transform(data[categorical_cols])
    encoded_data=pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols),index=data.index)

    # Drop the original categorical columns 
    data = data.drop(columns=categorical_cols)
    # Combine
    final_data = pd.concat([data, encoded_data], axis=1)

    # Align the final data with the training columns
    final_data = final_data.reindex(columns=training_columns, fill_value=0)
     
    # Scale only the numerical columns
    numerical_cols = [
        "Application order",
        "Age at enrollment",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)"
    ]

            
    final_data[numerical_cols] = scaler.transform(final_data[numerical_cols])
    
    # Ensure numeric 
    final_data=final_data.apply(pd.to_numeric, errors='coerce')
    final_data=final_data.fillna(0)
    
    # Predict probabilities
    dropout_index = list(model.classes_).index("Dropout")
    predictions = model.predict_proba(final_data)[:, dropout_index]


    # Add results to the original data
    original_data['Dropout Probability'] = predictions

    # Display the results
    st.success("Predictions generated successfully!")
    st.write(original_data)

    # Download button for the results
    st.download_button(
        "Download Predictions",
        original_data.to_csv(index=False),
        "predictions.csv"
    )