# Diabetes Prediction Streamlit App

This project is a Streamlit application for predicting diabetes based on user input. It utilizes a trained machine learning model to provide predictions and insights.

## Project Structure

```
diabetes-prediction-streamlit
├── src
│   ├── app.py          # Main application file for the Streamlit app
│   └── utils.py        # Utility functions for data preprocessing and model loading
├── best_diabetes_model.pkl  # Saved model file containing the trained diabetes prediction model
├── requirements.txt    # List of dependencies required for the project
└── README.md           # Documentation for the project
```

## Installation

To run this project, you need to have Python installed on your machine. Follow these steps to set up the environment:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd diabetes-prediction-streamlit
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit app, execute the following command in your terminal:

```
streamlit run src/app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Usage

Once the application is running, you can input the required features for diabetes prediction. The model will provide a prediction based on the input data.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.