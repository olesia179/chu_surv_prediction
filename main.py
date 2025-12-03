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

class PatientData(BaseModel):
    gender: str
    age: int
    BMI: float
    ICD9: int
    MOBPAT: str
    NRSF11: int
    OMS: int
    T: str
    N: str
    M: str
    DIA_MAX_TUMEUR_1: float
    DIA_MAX_TUMEUR_2: float
    DIA_MAX_TUMEUR_3: float
    DIA_MAX_TUMEUR_4: float
    CHIRURGIE_ANT_ON: int
    CHIMOANT_ON: int
    IRRA_ANT_ON: int
    IRRANT_OLOCA: int
    INSUFFURENALE_ON: int
    FUMEUROUINON: int
    DIABETIQUE_ON: int
    INSULO_ON: int
    PACEMAKER_ON: int
    ALLERGIE_ON: int
    DEGRURG: int
    TotalDose: int
    NbVolumes: int
    TotalSessions: int

app = FastAPI()

@app.post('/predict')
def predict(data : list[PatientData]):
    num = ['age', 'gender', 'BMI', 'DIA_MAX_TUMEUR_1', 'DIA_MAX_TUMEUR_2', 'DIA_MAX_TUMEUR_3', 'DIA_MAX_TUMEUR_4', 
                'INSUFFURENALE_ON', 'FUMEUROUINON', 'DIABETIQUE_ON', 'INSULO_ON', 'PACEMAKER_ON', 'ALLERGIE_ON', 
                'TotalDose', 'NbVolumes', 'TotalSessions', 'DEGRURG', 'CHIMOANT_ON', 'IRRA_ANT_ON', 'CHIRURGIE_ANT_ON']

    ohe = ['ICD9', 'IRRANT_OLOCA', 'MOBPAT', 'T', 'N', 'M']

    ord = ['NRSF11', 'OMS']

    data_list = [item.model_dump() for item in data]

    patients = pd.DataFrame(data_list)
    patients, _ = DataProcessor(num, ohe, ord).get_df(patients)

    output = { 'time_in_months' : 999 }
    if os.path.exists('./data/model.ubj'):
        model = xgb.Booster(model_file='./data/model.ubj')
        if os.path.exists('./data/preprocessor.pkl'):
            preprocessor = joblib.load('./data/preprocessor.pkl')
            patients = preprocessor.transform(patients)
                
        dnew = xgb.DMatrix(patients)
        result = model.predict(dnew)
        output = [{ 'time_in_months' : round(prediction) } for prediction in result.tolist()]
    else:
        raise FileNotFoundError("Model file not found. Please train the model first.")
    return output