
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(
    page_title="RAxT-Net Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ RAxT-Net")
st.subheader("Hybrid Cardiovascular Disease Prediction System")
st.write("RAxT-Net + XGBoost + TabNet")

st.info(
    "Academic machine-learning demonstration only. "
    "This result is not a medical diagnosis."
)

@st.cache_resource
def load_models():
    raxt = tf.keras.models.load_model("raxt_model.h5")
    extractor = tf.keras.models.load_model("raxt_feat_extractor.h5")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("model_xgb.json")

    tab_model = TabNetClassifier()
    tab_model.load_model("tabnet_model_checkpoint.zip")

    scaler = joblib.load("scaler.pkl")

    return raxt, extractor, xgb_model, tab_model, scaler

try:
    raxt, extractor, xgb_model, tab_model, scaler = load_models()
    st.success("All models loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.header("Enter Patient Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 52)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, 125)
        chol = st.number_input("Cholesterol", 50, 600, 212)
        fbs = st.selectbox("Fasting Blood Sugar (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])

    with col2:
        thalach = st.number_input("Maximum Heart Rate", 50, 250, 168)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Heart Disease")

if submitted:
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    values = [[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]]

    input_df = pd.DataFrame(values, columns=feature_names)

    try:
        scaled_input = scaler.transform(input_df)

        raxt_prob = float(
            raxt.predict(scaled_input, verbose=0).flatten()[0]
        )

        deep_features = extractor.predict(
            scaled_input, verbose=0
        )

        xgb_prob = float(
            xgb_model.predict_proba(deep_features)[0][1]
        )

        tab_prob = float(
            tab_model.predict_proba(
                scaled_input.astype(np.float32)
            )[0][1]
        )

        # Weights from notebook's saved-model inference cell
        final_prob = (
            0.35 * raxt_prob +
            0.35 * xgb_prob +
            0.30 * tab_prob
        )

        st.divider()
        st.header("Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("RAxT-Net", f"{raxt_prob * 100:.2f}%")
        c2.metric("XGBoost", f"{xgb_prob * 100:.2f}%")
        c3.metric("TabNet", f"{tab_prob * 100:.2f}%")

        st.subheader("Hybrid Ensemble Result")
        st.metric("Predicted Disease Probability",
                  f"{final_prob * 100:.2f}%")

        if final_prob > 0.5:
            st.error("Heart Disease: Positive prediction")
        else:
            st.success("Heart Disease: Negative prediction")

        st.caption(
            "Model output is experimental and must not be "
            "used to make medical decisions."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
