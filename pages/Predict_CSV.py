import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Survival Prediction", page_icon="📈")
st.markdown("# Survival Prediction")
st.sidebar.header("Predict survival")

# API_ENDPOINT = 'http://localhost:8000/predict'
API_ENDPOINT = 'https://chu-surv-prediction.onrender.com/predict'

uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv']
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    prediction = requests.post(url=API_ENDPOINT, json=df.to_dict(orient='records')).json()
    results_df = pd.DataFrame(prediction)
    st.markdown("## Prediction results:")
    results_df.insert(0, 'Patient_ID', df['numnat'])
    st.dataframe(results_df)