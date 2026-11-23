
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------
# LOAD PICKLE FILES
# --------------------------------

@st.cache_resource
def load_model_files():

    with open("heart_disease_ann_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("heart_disease_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("heart_disease_features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, preprocessor, features


try:
    model, preprocessor, features = load_model_files()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("❤️ Heart Disease Prediction System")

st.markdown(
    """
    ### Artificial Neural Network (ANN)

    This application uses a trained ANN model to predict
    the heart disease class from the entered information.
    """
)

st.divider()


# --------------------------------
# PATIENT INPUT FORM
# --------------------------------

st.subheader("Enter Patient Details")

with st.form("heart_disease_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=0,
            max_value=300,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200
        )

        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

    with col2:

        max_heart_rate = st.number_input(
            "Maximum Heart Rate",
            min_value=0,
            max_value=300,
            value=150
        )

        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-10.0,
            max_value=20.0,
            value=0.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            ["Up", "Flat", "Down"]
        )

        thal = st.selectbox(
            "Thal",
            ["Normal", "Fixed Defect", "Reversible Defect"]
        )

    submit = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )



# --------------------------------
# MAKE PREDICTION
# --------------------------------

if submit:

    input_values = {
        "age": age,
        "gender": gender,
        "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "slope": slope,
        "thal": thal
    }

    try:

        input_data = pd.DataFrame([input_values])

        # Match model training feature order
        input_data = input_data[features]

        # Apply preprocessing
        processed_data = preprocessor.transform(input_data)

        # Convert sparse matrix if needed
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        # ANN prediction
        probability = model.predict(
            processed_data,
            verbose=0
        )

        probability = float(
            np.asarray(probability).ravel()[0]
        )

        prediction = 1 if probability >= 0.5 else 0

        st.divider()
        st.subheader("Prediction Result")

        # Display raw model score
        st.write("Raw ANN Output:", probability)

        # Display prediction
        if prediction == 1:
            st.error("Model Prediction: Heart Disease")
        else:
            st.success("Model Prediction: No Heart Disease")

        st.metric(
            "ANN Output Score",
            f"{probability:.6f}"
        )

        # Display interpretation
        st.write(
            "Classification threshold: 0.50"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Heart Disease Prediction using Artificial Neural Networks."
)

st.warning(
    "Educational project only. This model is not clinically "
    "validated and must not be used for medical diagnosis "
    "or treatment decisions."
)
