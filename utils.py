# Utility functions for the diabetes prediction Streamlit app

import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """Load the trained diabetes prediction model from the specified path."""
    model = joblib.load(model_path)
    return model

def preprocess_input(data):
    """Preprocess the input data for prediction."""
    # Assuming data is a dictionary with feature names as keys
    df = pd.DataFrame(data, index=[0])
    
    # Replace zeros with NaN for specific columns if needed
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    
    # Impute missing values with the median
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())
    
    return df

def get_prediction(model, input_data):
    """Get the prediction from the model for the given input data."""
    prediction = model.predict(input_data)
    return prediction[0]  # Return the first prediction if input is a single instance