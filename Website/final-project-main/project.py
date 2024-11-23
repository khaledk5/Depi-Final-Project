import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
import plotly.express as px
import plotly.graph_objects as go

# Set up the app configuration
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling the application
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f3f4f6, #ffffff);
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ff5722;
            color: white;
        }
        .stButton > button {
            background-color: #6200ea;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 15px 25px;
            font-size: 18px;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #3700b3;
        }
        h1 {
            color: #6200ea;
            text-align: center;
            font-size: 36px;
        }
        h2, h3 {
            color: #ff5722;
        }
        .form-container {
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        .footer {
            font-size: 0.9em;
            text-align: center;
            padding: 20px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Main title of the app
st.title("🩺 Disease Predictor")

# Introduction message with details about the application
st.markdown("""
Welcome to the *Disease Predictor *! Select a dataset and enter the required values to determine if you're at risk of a particular disease.
""")

# Sidebar configuration for dataset selection
st.sidebar.header("Configuration")

# Options for datasets, each linked with an emoji for better UX
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Heart Disease ❤", "Brain Stroke 🧠", "Diabetes 🍭")
)

# Define the path to datasets (ensure these paths are correctly set to your files)
DATASETS = {
    "Heart Disease ❤": "HeartDiseaseML11.csv",
    "Brain Stroke 🧠": "brainstrokeML.csv",
    "Diabetes 🍭": "diabetes_1.csv"
}


@st.cache_resource
def load_data(name):
    """Load the selected dataset and cache it for efficiency."""
    try:
        data = pd.read_csv(DATASETS[name])
        return data
    except FileNotFoundError:
        st.error(f"Dataset file for {name} not found. Please check the file path.")
        return pd.DataFrame()


@st.cache_resource
def preprocess_data(df):
    """Clean and preprocess the dataset by handling missing values and encoding categorical data."""
    if df.empty:
        return None, None

    df = df.dropna()

    # Encode categorical columns for compatibility
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y, label_encoders


@st.cache_resource
def train_model(X, y):
    """Train a Random Forest Classifier and evaluate its accuracy."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return model, accuracy, X.columns.tolist()
    except Exception as e:
        st.error(f"Error in model training: {e}")
        return None, None, []


# Load and process the chosen dataset
df = load_data(dataset_name)
X, y, label_encoders = preprocess_data(df)

if X is not None and y is not None:
    model, accuracy, feature_names = train_model(X, y)

    if model:
        # Display the selected disease above the form
        st.subheader(f"Selected Disease: {dataset_name}")

        # Input form for prediction
        st.subheader("Enter Values to Predict the Outcome")
        with st.form(key='prediction_form', clear_on_submit=True):
            user_inputs = {}
            for feature in feature_names:
                if X[feature].dtype in [np.int64, np.float64]:
                    user_input = st.number_input(
                        label=f"Enter {feature}",
                        value=0.0,
                        format="%.2f",
                        step=0.1
                    )
                else:
                    unique_values = sorted(df[feature].unique())
                    user_input = st.selectbox(
                        label=f"Select {feature}",
                        options=unique_values,
                        index=0
                    )
                user_inputs[feature] = user_input

            # Predict button
            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            try:
                # Prepare user inputs for prediction
                input_data = pd.DataFrame([user_inputs])

                # Encode the categorical inputs
                for column in input_data.select_dtypes(include=['object', 'category']).columns:
                    if column in label_encoders:
                        input_data[column] = label_encoders[column].transform(input_data[column])

                # Ensure all features are present in the input data
                for feature in feature_names:
                    if feature not in input_data.columns:
                        input_data[feature] = 0

                # Get prediction and probabilities
                prediction_proba = model.predict_proba(input_data)[0]
                prediction = model.predict(input_data)
                outcome = "Infected" if prediction[0] == 1 else "Not Infected"

                # Display prediction outcome
                st.success(f"### Prediction Outcome: *{outcome}*")

                # Plot probability distribution with a bar chart
                fig_proba = go.Figure(data=[
                    go.Bar(
                        x=["Not Infected", "Infected"],
                        y=prediction_proba,
                        marker=dict(color=["#00C853", "#D32F2F"]),
                        text=[f"{prob:.2%}" for prob in prediction_proba],
                        textposition='auto'
                    )
                ])
                fig_proba.update_layout(
                    title="Prediction Probability Distribution",
                    xaxis_title="Outcome",
                    yaxis_title="Probability",
                    template="plotly_white"
                )
                st.plotly_chart(fig_proba)

            except NotFittedError:
                st.error("Model is not fitted yet.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        # Display model accuracy
        st.subheader("Model Performance")
        st.write(f"*Accuracy:* {accuracy:.2f}")

        # Display the dataset overview
        st.subheader(f"{dataset_name} Dataset Overview")
        st.dataframe(df.head())

        # Plot the distribution of a selected feature
        st.subheader(f"{dataset_name} Feature Distribution")
        feature_to_plot = st.selectbox("Select a feature to visualize", df.columns)
        fig = px.histogram(df, x=feature_to_plot, title=f"Distribution of {feature_to_plot}")
        st.plotly_chart(fig)

        # Footer information
        st.markdown("""
        ---
        ### About
        This application is a disease prediction tool created using *Streamlit*. Select a disease, enter the values, and receive real-time predictions!

        ### Developed by:
        - Metyas Monir Yousef
        - Khaled Ayman Farouk
        - Noor Eldeen Mohammed Shrief
        - Mostafa Rabea Hashem
 

        ### Supervised by:
        Dr. Moshera Ghallab
        """)
else:
    st.warning("Please select a valid dataset to proceed.")
