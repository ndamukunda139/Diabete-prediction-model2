import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_input
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# Set page configuration
st.set_page_config(
    page_title="Diabete risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1d3557;
        margin-bottom: 0.5rem;
    }
    .info-text {
        background-color: #f1faee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #a8dadc;
        color: #1d3557;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        color: #856404;
    }
    .severe-box {
        background-color: #ff2000;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        color: #000000;
    }
    .center {
        margin: auto;
        width: 50%;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
        padding: 10px;
    }
    .stButton>button {
        background-color: #457b9d;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d3557;
    }
    /* Improve general text visibility */
    p, h1, h2, h3, label {
        color: #1d3557;
    }
    /* Improve form field visibility */
    .stNumberInput, .stSelectbox {
        background-color: #f1faee !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained model from disk"""
    try:
        with open('best_diabetes_model.pkl', 'rb') as file:
            model = joblib.load('best_diabetes_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'best_diabetes_model.pkl' is in the current directory.")
        return None
# Preprocess input data
@st.cache_data
def preprocess_input(input_data):
    """Preprocess the input data for prediction"""
    # Ensure the input data is in the correct format
    if isinstance(input_data, pd.DataFrame):
        # Convert to numeric and handle NaN values
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        return input_data
    else:
        st.error("Input data must be a pandas DataFrame.")
        return None
    
# Laod feature info 
@st.cache_data
def load_feature_info():
    """"Return information about features for prediction form"""
    return {
        'Pregnancies': {'type': 'number', 'min': 0, 'max': 15, 'help':"Enter number of times pregnant"},
        'Glucose':{'type' : 'number', 'min': 40, 'max': 200, 'help' : "Enter Plasma glucose concentration a 2 hours in an oral glucose tolerance test"},
        'BloodPressure': {'type' : 'number', 'min' : 30, 'max':100, 'help': "Enter Diastolic blood pressure (mm Hg)"},
        'SkinThickness':{'type': 'number', 'min': 10, 'max': 45, 'help': "Enter triceps skin fold thickness (mm)"},
        'Insulin': {'type': 'number', 'min': 100, 'max': 150, 'help': "Enter 2-Hour serum insulin (mu U/ml)"},
        'BMI': {'type':'number', 'min': 15, 'max' : 60, 'help': "Body mass index (weight in kg/(height in m)^2)"},
        'DiabetesPedigreeFunction': {'type': 'number', 'min': 0.070, 'max': 1.5, 'help': "Enter Diabetes pedigree function"},
        'Age': {'type': 'number', 'min': 1, 'max': 100, 'help': "Enter Age (years)"}
    }
# Generate a random data function
def generate_random_patient():
    """Generate random but realistic patient data"""
    import random

    # Get feature info to use the appropriate options
    feature_info = load_feature_info()

    # Generate random patient data
    random_patient = {
        'Pregnancies': random.randint(feature_info['Pregnancies']['min'], feature_info['Pregnancies']['max']),
        'Glucose': random.randint(feature_info['Glucose']['min'], feature_info['Glucose']['max']),
        'BloodPressure': random.randint(feature_info['BloodPressure']['min'], feature_info['BloodPressure']['max']),
        'SkinThickness': random.randint(feature_info['SkinThickness']['min'], feature_info['SkinThickness']['max']),
        'Insulin': random.randint(feature_info['Insulin']['min'], feature_info['Insulin']['max']),
        'BMI': random.uniform(feature_info['BMI']['min'], feature_info['BMI']['max']),
        'DiabetesPedigreeFunction': random.uniform(feature_info['DiabetesPedigreeFunction']['min'], feature_info['DiabetesPedigreeFunction']['max']),
        'Age': random.randint(feature_info['Age']['min'], feature_info['Age']['max'])
    }    
    return random_patient

# Generate a random sample data function
def generate_sample_data():
    """Generate a random sample data for testing"""
    import random

    # Get feature info to use the appropriate options
    sample_size = 100
    feature_info = load_feature_info()

    sample_data = {
        'Pregnancies': [random.randint(0, 15) for _ in range(sample_size)],
        'Glucose': [random.randint(40, 200) for _ in range(sample_size)],
        'BloodPressure': [random.randint(30, 100) for _ in range(sample_size)] ,
        'SkinThickness': [random.randint(10, 45) for _ in range(sample_size)],
        'Insulin': [random.randint(100, 150) for _ in range(sample_size)],
        'BMI': [random.uniform(15.0, 60.0) for _ in range(sample_size)],
        'DiabetesPedigreeFunction': [random.uniform(0.070, 1.5) for _ in range(sample_size)],
        'Age': [random.randint(1, 100) for _ in range(sample_size)],
        'Outcome': [random.choice([1, 0]) for _ in range(sample_size)]
    }

    # Create a DataFrame from the sample data
    df = pd.DataFrame(sample_data)
    return df

# Preprocess input data
def preprocess_input(input_data, model):
    """Process input data to match model's expected format"""
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Ensure all columns are present in the input DataFrame
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]   

    return input_df



# Get prediction probabilities function
def get_prediction_probabilities(model, input_df):
    """Get prediction and probability from the model"""

    # Get prediction
    prediction = model.predict(input_df)[0]

    # Get probabilities for each class
    probabilities = model.predict_proba(input_df)[0]

    # Get prediction and probability from the model
    results = {
      
        'prediction': prediction,
        'probabilities_diabete': probabilities[1],  # Probability of diabete
        'probabilities_no_diabete': probabilities[0]  # Probability of no diabete
    }
    return results

# Display feature importance
def display_feature_importance(model, input_df):
    """Display feature importance for the prediction"""
    st.markdown("<div class='sub-header'>Feature Importance Analysis</div>", unsafe_allow_html=True)

    # Check if model is a pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        feature_importance = None

        # Process input data through the preprocessing pipeline
        processed_input = model['preprocessor'].transform(input_df)

        if hasattr(model['model'], 'feature_importances_'):
            feature_importance = model['model'].feature_importances_
        elif hasattr(model['model'], 'coef_'):
            feature_importance = model['model'].coef_[0]

        if feature_importance is not None:
            # Try to get feature names
            try:
                feature_names = model['preprocessor'].get_feature_names_out()
            except:
                # If can't get names, create generic ones
                feature_names = [f"Feature {i}" for i in range(len(feature_importance))]

            # Create a DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(20)

            # Plot feature importance
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title='Top 10 Feature Importance',
                        color='Importance', color_continuous_scale='Viridis')

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model does not provide feature importance.")
                

# Display eda visualizations
def display_eda_visualizations(sample_data):
    """Display EDA visualizations for the sample data"""
    #List of tabs in EDA section
    st.write("### EDA Overview")
    st.write("This section provides visualizations to explore the dataset and understand the relationships between features.")  
    st.write("You can select different features to visualize their distributions, correlations, and how they relate to diabete status.")
    st.write("Use the tabs below to navigate through different EDA perspectives.")

    # Generate sample data for visualizations
    df = generate_sample_data()

    # Create tabs for different EDA perspectives
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distributions", "Correlations", "Diabetes Factors"])
    with eda_tab1:
        st.markdown("<div class='sub-header'>Distributions of Features</div>", unsafe_allow_html=True)

        st.write("### Feature Distributions")
        st.write("This section shows the distribution of each feature in the dataset.")

        # Feature selection for distribution
        dist_feature = st.selectbox(
            "Select a feature to display its distribution",
            options=df.columns.tolist(),
            help="Choose a feature to visualize its distribution"
        )
        fig = px.histogram(
            df, x=dist_feature, color='Outcome', barmode='overlay',
            marginal='box', opacity=0.7,
            color_discrete_map={'No diabete': '#28a745', 'Diabete': '#dc3545'},
            title=f"Distribution of {dist_feature} by diabete Status"
        )
        st.plotly_chart(fig, use_container_width=True)

    with eda_tab2:
        st.markdown("<div class='sub-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
        st.markdown("Explore relationships between different features.")

        # Scatter plot
        x_feature = st.selectbox(
            "Select X-axis feature",
            options=df.columns.tolist(),
            help="Choose a feature for the X-axis"
        )
        y_feature = st.selectbox(
            "Select Y-axis feature",
            options=df.columns.tolist(),
            help="Choose a feature for the Y-axis"
        )
        fig = px.scatter(
            df, x=x_feature, y=y_feature, color='Outcome',
            color_discrete_map={'No diabete': '#28a745', 'Diabete': '#dc3545'},
            opacity=0.7, title=f"Correlation between {x_feature} and {y_feature}",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show correlation heatmap
        st.markdown("### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int', 'float'])
        corr = numeric_df.corr()

        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale='RdBu_r', title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    with eda_tab3:
        st.markdown("<div class='sub-header'>Diabete Factors</div>", unsafe_allow_html=True)
        st.write("### Survival Factors Analysis")
        st.write("This section can include survival analysis based on the features.")
        st.write("Explore how different factors contribute to diabete status.")
        # Example: Bar chart of diabete status by age
        age_bins = pd.cut(df['Age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
        age_distribution = df.groupby(age_bins)['Outcome'].value_counts().unstack().fillna(0)
        age_distribution = age_distribution.rename(columns={0: 'No diabete', 1: 'Diabete'})
        age_distribution = age_distribution.reset_index()
        fig = px.bar(
            age_distribution, x='Age', y=['No diabete', 'Diabete'],
            title="Diabete Status by Age Group",
            color_discrete_map={'No diabete': '#28a745', 'Diabete': '#dc3545'},
            labels={'value': 'Count', 'variable': 'Diabete Status'}
        )
        st.plotly_chart(fig, use_container_width=True)  

        # Example: Box plot of glucose levels by diabete status
        st.write("### Glucose Levels by Diabete Status")
        fig = px.box(
            df, x='Outcome', y='Glucose',
            color='Outcome', color_discrete_map={'No diabete': '#28a745', 'Diabete': '#dc3545'},
            title="Glucose Levels by Diabete Status"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Example: Bar chart of diabete status by BMI
        st.write("### BMI Distribution by Diabete Status")
        bmi_bins = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, 100], 
                          labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III'])
        bmi_distribution = df.groupby(bmi_bins)['Outcome'].value_counts().unstack().fillna(0)
        bmi_distribution = bmi_distribution.rename(columns={0: 'No diabete', 1: 'Diabete'})
        bmi_distribution = bmi_distribution.reset_index()
        fig = px.bar(
            bmi_distribution, x='BMI', y=['No diabete', 'Diabete'],
            title="Diabete Status by BMI Category",
            color_discrete_map={'No diabete': '#28a745', 'Diabete': '#dc3545'},
            labels={'value': 'Count', 'variable': 'Diabete Status'}
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit app"""
    st.markdown("<h1 class='main-header'>Diabete disease Prediction</h1>", unsafe_allow_html=True)

    # Create tabs for different app sections
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "EDA", "Information", "About"])

    with tab1:
        st.markdown("<div class='sub-header'>Patient Information</div>", unsafe_allow_html=True)

        # Load model
        model = load_model()
        if model is None:
            st.error("Model could not be loaded. Please check the model file.")
            return
        # Load feature info
        feature_info = load_feature_info()

        # Add sample patient button outside the form
        if st.button("Load Random Sample Patient"):
            st.session_state.input_data = generate_random_patient()
            st.rerun()

        # Create form for user input
        with st.form("prediction_form"):
            st.write("### Enter Patient Details")
            input_data = {}
            for feature, info in feature_info.items():
                if info['type'] == 'number':
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=info['min'],
                        max_value=info['max'],
                        help=info['help']
                    )
                elif info['type'] == 'select':
                    input_data[feature] = st.selectbox(
                        feature,
                        options=info['options'],
                        help=info['help']
                    )
            # Submit button
            submit_button = st.form_submit_button("Predict")

            if submit_button:
                with st.spinner("Processing..."):
                    # Preprocess input data
                    processed_data = preprocess_input(pd.DataFrame([input_data]), model)
                    if processed_data is None:
                        st.error("Error in preprocessing input data.")
                        return

                    # Get prediction probabilities
                    results = get_prediction_probabilities(model, processed_data)

                    # Display result
                    st.markdown("<div class='sub-header'>Prediction Result</div>", unsafe_allow_html=True)

                    # Create a two-column layout for the results
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if results['prediction'] == 0:
                            st.markdown(f"""
                                <div class='success-box'>
                                    <h3>Prediction: No Diabete</h3>
                                    <p>The model predicts that the patient is likely not have diabete.</p>
                                    <p>Confidence: {results['probabilities_no_diabete']:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        elif results['prediction'] == 1:
                            st.markdown(f"""
                                <div class='severe-box'>
                                    <h3>Prediction: Diabete </h3>
                                    <p>The model predicts that the patient is likely to have severe diabete.</p>
                                    <p>Confidence: {results['probabilities_diabete']:.2%}</p>   
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class='warning-box'>
                                    <h3> Error: Unable to make a prediction</h3>
                                    <p>Please check the input data and try again.</p>
                                </div>
                                """, unsafe_allow_html=True)
                    with col2:   
                        # Create a gauge chart showing probability with improved colors
                        st.markdown("<div class='center', >Prediction Probability Gauge</div>", unsafe_allow_html=True)

                        import plotly.graph_objects as go

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=float(results['probabilities_diabete']),
                            title={'text': "Probability of Diabetes", 'font': {'color': '#1d3557'}},
                            gauge={
                                'axis': {'range': [0, 1], 'tickcolor': "#1d3557", 'tickwidth': 2, 'ticklen': 10},
                                'bar': {'color': "rgba(0, 0, 0, 0)"},
                                'steps': [
                                    {'range': [0, 0.3], 'color': "#a8dadc"},
                                    {'range': [0.3, 0.7], 'color': "#457b9d"},
                                    {'range': [0.7, 1], 'color': "#e63946"}
                                ],
                                'threshold': {
                                    'line': {'color': "#1d3557", 'width': 4},
                                    'thickness': 0.75,
                                    'value': float(results['probabilities_diabete'])
                                }
                            }
                        ))
                        fig.update_layout(height=250, font={'color': '#1d3557'})
                        st.plotly_chart(fig, use_container_width=True)

                    # Display feature importance
                    display_feature_importance(model, processed_data)

                    # Display warning disclaimer
                    st.markdown("""
                    <div class='info-text'>
                        <p><strong>Disclaimer:</strong> This prediction is based on a machine learning model and should be
                        used for informational purposes only. Always consult with healthcare professionals for
                        medical decisions.</p>
                    </div>
                """, unsafe_allow_html=True)
                    

    with tab2:
        st.markdown("<div class='sub-header'>Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)
        # Display EDA visualizations
        display_eda_visualizations(generate_sample_data())
    with tab3:
        st.markdown("<div class='sub-header'>Information</div>", unsafe_allow_html=True )
        st.write("### About diabetes prediction model")
        st.write("""
        This model predicts the likelihood of diabetes based on various health parameters. It uses a machine learning
        algorithm trained on historical patient data to provide predictions. The model considers factors such as glucose levels,
        blood pressure, BMI, and age to assess the risk of diabetes.
        """)
        st.write("### How to use this app")
        st.write("""
        1. Enter the patient's health details in the form provided.
        2. Click on the 'Predict' button to get the prediction.
        3. The app will display the prediction result along with the probability of diabetes.
        4. You can also view exploratory data analysis (EDA) visualizations to understand the   data better.
        """)
        st.write("### Disclaimer")
        st.write("""
        This app is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a healthcare professional for medical decisions.
        """)
    with tab4:
        st.write("### About this App")
        st.write("""        This app is designed to predict the likelihood of diabetes based on various health parameters. It uses a machine learning model trained on historical patient data to provide predictions. The app allows users to enter patient details and get predictions along with probabilities. It also includes exploratory data analysis (EDA) visualizations to help users understand the data better.""")
        st.write("### Model Information")
        st.write("""    The model used in this app is a machine learning model trained on a dataset of diabetes patients. It considers various health parameters such as glucose levels, blood pressure, BMI, and age to assess the risk of diabetes. The model provides predictions along with probabilities for each class (Diabetes and No Diabetes).""")
        st.write("### Contact Information")
        st.write("""        For any questions or feedback regarding this app, please contact the developer at [diabeteprediction@gmail.com](mailto:diabeteprediction@gmail.com).""")   
        st.write("### Acknowledgements")
        st.write("""        This app is built using Streamlit and leverages machine learning techniques for diabetes prediction. The model is trained on a dataset of diabetes patients and uses various health parameters to provide predictions. Special thanks to the contributors and the open-source community for their support in building this app.""")

if __name__ == "__main__":
    main()
    # Run the main function to start the app
    # This will initialize the Streamlit app and display the UI components  

