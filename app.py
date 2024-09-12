import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import time

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define function to make predictions
def predict_breast_cancer(data):
    prediction = model.predict(data)
    return prediction

# Streamlit app
st.title("Breast Cancer Diagnosis App")
st.write("## Using Machine Learning to Predict Breast Cancer")
st.write("This app predicts whether a tumor is benign or malignant based on inputted medical measurements.")

# Add medical image (CT Scan or any medical-related image)
st.sidebar.header("Medical Instruments")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/CTscanner.jpg/250px-CTscanner.jpg", 
                 caption="CT Scanner", use_column_width=True)

# Collect user input for all the features
st.sidebar.header("Input Features")

# Spinning loader when processing
with st.spinner("Loading input form..."):
    id = st.sidebar.text_input("Patient ID", "842302")
    diagnosis = st.sidebar.selectbox("Diagnosis", ["M", "B"])  # M for Malignant, B for Benign

    radius_mean = st.sidebar.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=17.99)
    texture_mean = st.sidebar.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=10.38)
    perimeter_mean = st.sidebar.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=122.8)
    area_mean = st.sidebar.number_input("Area Mean", min_value=0.0, max_value=2000.0, value=1001.0)  # Changed to 1001.0
    smoothness_mean = st.sidebar.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.1184)
    compactness_mean = st.sidebar.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.2776)
    concavity_mean = st.sidebar.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.3001)
    concave_points_mean = st.sidebar.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.1471)
    symmetry_mean = st.sidebar.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.2419)
    fractal_dimension_mean = st.sidebar.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0, value=0.07871)

    radius_se = st.sidebar.number_input("Radius SE", min_value=0.0, max_value=10.0, value=1.095)
    texture_se = st.sidebar.number_input("Texture SE", min_value=0.0, max_value=10.0, value=0.9053)
    perimeter_se = st.sidebar.number_input("Perimeter SE", min_value=0.0, max_value=20.0, value=8.589)
    area_se = st.sidebar.number_input("Area SE", min_value=0.0, max_value=500.0, value=153.4)
    smoothness_se = st.sidebar.number_input("Smoothness SE", min_value=0.0, max_value=1.0, value=0.006399)
    compactness_se = st.sidebar.number_input("Compactness SE", min_value=0.0, max_value=1.0, value=0.04904)
    concavity_se = st.sidebar.number_input("Concavity SE", min_value=0.0, max_value=1.0, value=0.05373)
    concave_points_se = st.sidebar.number_input("Concave Points SE", min_value=0.0, max_value=1.0, value=0.01587)
    symmetry_se = st.sidebar.number_input("Symmetry SE", min_value=0.0, max_value=1.0, value=0.03003)
    fractal_dimension_se = st.sidebar.number_input("Fractal Dimension SE", min_value=0.0, max_value=1.0, value=0.006193)

    radius_worst = st.sidebar.number_input("Radius Worst", min_value=0.0, max_value=50.0, value=25.38)
    texture_worst = st.sidebar.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=17.33)
    perimeter_worst = st.sidebar.number_input("Perimeter Worst", min_value=0.0, max_value=300.0, value=184.6)
    area_worst = st.sidebar.number_input("Area Worst", min_value=0.0, max_value=3000.0, value=2019.0)  # Changed to 2019.0
    smoothness_worst = st.sidebar.number_input("Smoothness Worst", min_value=0.0, max_value=1.0, value=0.1622)
    compactness_worst = st.sidebar.number_input("Compactness Worst", min_value=0.0, max_value=1.0, value=0.6656)
    concavity_worst = st.sidebar.number_input("Concavity Worst", min_value=0.0, max_value=1.0, value=0.7119)
    concave_points_worst = st.sidebar.number_input("Concave Points Worst", min_value=0.0, max_value=1.0, value=0.2654)
    symmetry_worst = st.sidebar.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.4601)
    fractal_dimension_worst = st.sidebar.number_input("Fractal Dimension Worst", min_value=0.0, max_value=1.0, value=0.1189)

# Store input values in a list
input_features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, 
                  concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, 
                  texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, 
                  symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, 
                  smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, 
                  fractal_dimension_worst]

# Convert the input into a numpy array and reshape for prediction
input_data = np.array(input_features).reshape(1, -1)

# Add a button to trigger the prediction
if st.sidebar.button('Predict'):
    with st.spinner("Making prediction..."):
        # Simulate a delay
        time.sleep(2)
        
        # Scale the input data (assuming the model was trained on scaled data)
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make predictions
        prediction = predict_breast_cancer(input_data_scaled)

        # Display predictions
        st.write("Prediction:")
        if prediction[0] == 1:
            st.error("The model predicts the tumor is Malignant.")
        else:
            st.success("The model predicts the tumor is Benign.")

# Add additional educational content in the sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a machine learning model trained on the Breast Cancer Wisconsin dataset to predict "
    "the likelihood of a tumor being malignant or benign based on medical features."
)
