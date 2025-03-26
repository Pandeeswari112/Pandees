# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:10:56 2025

@author: pande
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Function to preprocess the data (you can adjust this based on your dataset)
def preprocess_data(data):
    # Fill missing values if necessary (more robust handling)
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].median(), inplace=True)  # Fill with median for numeric columns
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)  # Fill with mode for non-numeric
    # Encode string columns as integers
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    return data, label_encoders

# Function to create and train a simple RandomForest model for attack detection
def create_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit app
st.title('DDoS Attack Detection Dashboard')

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ['Upload Data', 'Analyze Data', 'Train Model', 'Evaluate Model', 'Make Prediction', 'Logout'])

if options == 'Upload Data':
    st.subheader('Upload your traffic data')
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                data = pd.read_csv(stringio, delimiter="\t")  # Assuming tab-separated values
            st.write("### Uploaded Data Preview:")
            st.dataframe(data.head())

            # Preprocess the data
            data, label_encoders = preprocess_data(data)
            st.write("Data preprocessed successfully.")

            # Save the preprocessed data in session state for further use
            st.session_state['data'] = data
            st.session_state['label_encoders'] = label_encoders
        except Exception as e:
            st.error(f"Error processing file: {e}")

if options == 'Analyze Data' and 'data' in st.session_state:
    st.subheader('Data Analysis')
    data = st.session_state['data']
    st.write("Data description:")
    st.dataframe(data.describe())
    st.write("Correlation Matrix:")
    corr_matrix = data.corr()
    st.dataframe(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

if options == 'Train Model' and 'data' in st.session_state:
    st.subheader('Train a Model')
    data = st.session_state['data']

    # Display the column names to guide the user
    st.write("Columns available for prediction:", data.columns)

    # Select the target column (e.g., 'Attack' or similar)
    target_column = st.selectbox("Select the target column (e.g., 'Attack' or 'Label')", data.columns)

    # Selecting feature columns (the rest of the columns except the target)
    feature_columns = [col for col in data.columns if col != target_column]

    X = data[feature_columns]
    y = data[target_column]

    # Optionally, scale the features
    scale_data = st.checkbox("Scale features")
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split data into training and testing sets
    st.subheader('Splitting Data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write('Data has been split into training and testing sets.')

    # Train the model
    model = create_model(X_train, y_train)
    st.write('Model has been trained.')

    # Save the model and other information in session state for further use
    st.session_state['model'] = model
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['scaler'] = scaler if scale_data else None
    st.session_state['feature_columns'] = feature_columns

if options == 'Evaluate Model' and 'model' in st.session_state:
    st.subheader('Evaluate the Model')
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Model accuracy: {accuracy:.2f}')

    # Show classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write('Classification Report:')
    st.dataframe(pd.DataFrame(report).transpose())

if options == 'Make Prediction' and 'model' in st.session_state:
    st.subheader('Make a New Prediction')
    model = st.session_state['model']
    feature_columns = st.session_state['feature_columns']
    scaler = st.session_state['scaler']
    data = st.session_state['data']  # Retrieve data to access feature columns

    # Collect user input for making a new prediction
    input_data = {}
    for column in feature_columns:
        # More robust input handling: try/except and type checking
        while True:
            try:
                if pd.api.types.is_numeric_dtype(data[column]):
                    input_data[column] = st.number_input(f'Enter value for {column}')
                else:  # For non-numeric features, provide a text input
                    input_data[column] = st.text_input(f'Enter value for {column}')
                break  # Exit loop if input is valid
            except ValueError:
                st.error("Invalid input. Please enter a valid value.")

    # Convert input data to a DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Scale input data if necessary (IMPORTANT: transform, not fit_transform)
    if scaler is not None:
        input_df = scaler.transform(input_df)

    # Predict the class (DoS or Normal)
    if st.button('Make Prediction'):
        prediction = model.predict(input_df)
        prediction_label = "DDoS Attack" if prediction[0] == 1 else "Normal Traffic"
        st.write(f'Prediction: {prediction_label}')

if options == 'Logout':
    st.subheader('You have been logged out.')
    # Clear all session state
    for key in st.session_state.keys():
        del st.session_state[key]

    st.write("You have successfully logged out.")
