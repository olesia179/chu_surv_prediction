from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import streamlit as st
import os
import xgboost as xgb
from modules.processor.DataProcessor import DataProcessor

st.set_page_config(
    page_title="Survival Prediction",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a page above.")

class request_body(BaseModel):
    gender: str
    age: int
    BMI: float
    ICD9: int
    mobility: str
    NRSF11: int
    OMS: int
    T: int
    N: int
    M: int
    dia_max_tumeur_1: float
    dia_max_tumeur_2: float
    dia_max_tumeur_3: float
    dia_max_tumeur_4: float
    chirurgie: int
    chimoterapy: int
    irradiation: int
    irradiation_location: int
    kidney_failure: int
    smoker: int
    diabetic: int
    insulin: int
    pacemaker: int
    allergies: int
    emergent: int
    totalDose: int
    nbVolumes: int
    totalSessions: int

app = FastAPI()

@app.post('/predict')
def predict(data : request_body):
    num = ['age', 'gender', 'BMI', 'DIA_MAX_TUMEUR_1', 'DIA_MAX_TUMEUR_2', 'DIA_MAX_TUMEUR_3', 'DIA_MAX_TUMEUR_4', 
                'INSUFFURENALE_ON', 'FUMEUROUINON', 'DIABETIQUE_ON', 'INSULO_ON', 'PACEMAKER_ON', 'ALLERGIE_ON', 
                'TotalDose', 'NbVolumes', 'TotalSessions', 'DEGRURG', 'CHIMOANT_ON', 'IRRA_ANT_ON', 'CHIRURGIE_ANT_ON']

    ohe = ['ICD9', 'IRRANT_OLOCA', 'MOBPAT', 'T', 'N', 'M']

    ord = ['NRSF11', 'OMS']


    patient = pd.DataFrame([{
            'gender': data.gender,
            'age': data.age,
            'BMI': data.BMI,
            'ICD9': data.ICD9,
            'MOBPAT': data.mobility,
            'NRSF11': data.NRSF11,
            'OMS': data.OMS,
            'T': data.T,
            'N': data.N,
            'M': data.M,
            'DIA_MAX_TUMEUR_1': data.dia_max_tumeur_1,
            'DIA_MAX_TUMEUR_2': data.dia_max_tumeur_2,
            'DIA_MAX_TUMEUR_3': data.dia_max_tumeur_3,
            'DIA_MAX_TUMEUR_4': data.dia_max_tumeur_4,
            'CHIRURGIE_ANT_ON': data.chirurgie,
            'CHIMOANT_ON': data.chimoterapy,
            'IRRA_ANT_ON': data.irradiation,
            'IRRANT_OLOCA': data.irradiation_location,
            'INSUFFURENALE_ON': data.kidney_failure,
            'FUMEUROUINON': data.smoker,
            'DIABETIQUE_ON': data.diabetic,
            'INSULO_ON': data.insulin,
            'PACEMAKER_ON': data.pacemaker,
            'ALLERGIE_ON': data.allergies,
            'DEGRURG': data.emergent,
            'TotalDose': data.totalDose,
            'NbVolumes': data.nbVolumes,
            'TotalSessions': data.totalSessions
            }])
    patient, _ = DataProcessor(num, ohe, ord).get_df(patient)
    
    prediction = 999
    if os.path.exists('./data/model.ubj'):
        model = xgb.Booster(model_file='./data/model.ubj')
        if os.path.exists('./data/preprocessor.pkl'):
            preprocessor = joblib.load('./data/preprocessor.pkl')
            patient = preprocessor.transform(patient)
                
        dnew = xgb.DMatrix(patient)
        result = model.predict(dnew)
        if result.size > 0:
            prediction = result[0].item()
    else:
        raise FileNotFoundError("Model file not found. Please train the model first.")
    return { 'time_in_months' : round(prediction) }