
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(
    page_title="Heart Disease ML Demo",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Prediction")
st.write(
    "An educational machine-learning demonstration "
    "using the UCI Heart Disease dataset."
)

st.warning(
    "This application is for educational purposes only. "
    "It is not a medical diagnostic tool and must not be "
    "used to make treatment decisions."
)

if not MODEL_PATH.exists():
    st.error(
        "model.pkl was not found. Run `python train_model.py` first."
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter the dataset features")

with st.form("heart_disease_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=50
        )
        sex = st.selectbox(
            "Sex code", [0, 1],
            format_func=lambda v: {
                0: "0 — Female",
                1: "1 — Male",
            }[v],
        )
        cp = st.selectbox(
            "Chest pain type",
            [1, 2, 3, 4],
        )
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50, max_value=250, value=120,
        )
        chol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=50, max_value=700, value=200,
        )

    with col2:
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            [0, 1],
        )
        restecg = st.selectbox(
            "Resting ECG category",
            [0, 1, 2],
        )
        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=50, max_value=250, value=150,
        )
        exang = st.selectbox(
            "Exercise-induced angina",
            [0, 1],
        )
        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0, max_value=10.0,
            value=1.0, step=0.1,
        )

    with col3:
        slope = st.selectbox(
            "Peak exercise ST slope",
            [1, 2, 3],
        )
        ca = st.selectbox(
            "Major vessels (ca)",
            [0, 1, 2, 3],
        )
        thal = st.selectbox(
            "Thal category",
            [3, 6, 7],
        )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    try:
        prediction = int(model.predict(input_data)[0])

        st.divider()
        st.subheader("Model output")

        if prediction == 1:
            st.warning(
                "The model classified the input as "
                "the positive class."
            )
        else:
            st.success(
                "The model classified the input as "
                "the negative class."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            positive_index = list(model.classes_).index(1)
            score = float(probabilities[positive_index])

            st.metric(
                "Model-estimated positive-class score",
                f"{score:.1%}",
            )

        st.caption(
            "The output reflects this trained model and dataset. "
            "It is not an individual's clinical risk estimate "
            "or a medical diagnosis."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
