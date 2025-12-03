import streamlit as st
from modules.loader.DataLoader import DataLoader
from modules.processor.DataProcessor import DataProcessor
import pandas as pd
import requests

st.set_page_config(page_title="Survival Prediction", page_icon="📈")

st.markdown("# Survival Prediction")
st.sidebar.header("Predict survival")

# API_ENDPOINT = 'http://localhost:8000/predict'
API_ENDPOINT = 'https://chu-surv-prediction.onrender.com/predict'

def get_values_from_df(column_name, df) :
    """
    Get the list of unique values in a column of a dataframe.
    """
    return df[column_name].dropna().sort_values().unique()

@st.cache_data
def init() -> pd.DataFrame:
    data_loader = DataLoader()
    df_raw = data_loader.load_from_csv()
    data_processor = DataProcessor()
    df_raw = data_processor.fix_types(df_raw)
    return df_raw

def main():
    df = init()

    gender_options = ["Female", "Male"]
    gender = st.pills("Gender", gender_options, selection_mode="single", default = "Female")

    age = st.number_input('Age', min_value = 0, max_value = 100, value = 30)

    bmi = st.number_input('BMI', min_value = 5.00, max_value = 50.00, value = 22.00)

    icd9 = st.selectbox('ICD9', get_values_from_df('ICD9', df), index = None)
    mobpat = st.selectbox('Mobility mean', get_values_from_df('MOBPAT', df), index = None)
    nrsf11 = st.selectbox('Nutritional Risk Screening (NRSF11)', get_values_from_df('NRSF11', df), index = None)
    oms = st.selectbox('ECOG score', get_values_from_df('OMS', df), index = None)
    t = st.selectbox('T', get_values_from_df('T', df), index = None)
    n = st.selectbox('N', get_values_from_df('N', df), index = None)
    m = st.selectbox('M', get_values_from_df('M', df), index = None)

    dia_max_tumeur_1 = st.number_input('Tumeur 1 (max diamètre)', min_value = 0, max_value = 100, value = 1)
    dia_max_tumeur_2 = st.number_input('Tumeur 2 (max diamètre)', min_value = 0, max_value = 100, value = 0)
    dia_max_tumeur_3 = st.number_input('Tumeur 3 (max diamètre)', min_value = 0, max_value = 100, value = 0)
    dia_max_tumeur_4 = st.number_input('Tumeur 4 (max diamètre)', min_value = 0, max_value = 100, value = 0)

    chirurgie = st.checkbox('Previous surgery')
    chimoterapy = st.checkbox('Previous chemotherapy')
    irradiation = st.checkbox('Previous irradiation')
    irrad_loc_map = {
        'Unknown': -1,
        'Other body site': 1,
        'Same body site': 2
    }
    irrad_loc_options = irrad_loc_map.keys()
    irradiation_location = st.pills("Previous irradiation type", irrad_loc_options, selection_mode="single", default = 'Unknown')
    if irradiation_location:
        irradiation_location = irrad_loc_map[irradiation_location]

    kidney_failure = st.checkbox('Kidney failure')
    smoker = st.checkbox('Smoker')
    diabetic = st.checkbox('Diabetic')
    insulin = st.checkbox('Requiring insulin')
    pacemaker = st.checkbox('Has pacemaker')
    allergies = st.checkbox('Has allergies')
    emergent = st.checkbox('Emergent')

    totalDose = st.number_input('Total dose', min_value = 0, max_value = 20000, value = 1000)
    nbVolumes = st.number_input('Number of treatment volumes', min_value = 1, max_value = 10, value = 1)
    totalSessions = st.number_input('All volumes sessions', min_value = 1, max_value = 15, value = 1)

    prediction = ''
    _, col2, _ = st.columns(3)
    with col2 :
        if col2.button('Predict') :
            data = {'gender': gender,
                    'age': age,
                    'BMI': bmi,
                    'ICD9': icd9,
                    'MOBPAT': mobpat,
                    'NRSF11': nrsf11,
                    'OMS': oms,
                    'T': t,
                    'N': n,
                    'M': m,
                    'DIA_MAX_TUMEUR_1': dia_max_tumeur_1,
                    'DIA_MAX_TUMEUR_2': dia_max_tumeur_2,
                    'DIA_MAX_TUMEUR_3': dia_max_tumeur_3,
                    'DIA_MAX_TUMEUR_4': dia_max_tumeur_4,
                    'CHIRURGIE_ANT_ON': chirurgie,
                    'CHIMOANT_ON': chimoterapy,
                    'IRRA_ANT_ON': irradiation,
                    'IRRANT_OLOCA': irradiation_location,
                    'INSUFFURENALE_ON': kidney_failure,
                    'FUMEUROUINON': smoker,
                    'DIABETIQUE_ON': diabetic,
                    'INSULO_ON': insulin,
                    'PACEMAKER_ON': pacemaker,
                    'ALLERGIE_ON': allergies,
                    'DEGRURG': emergent,
                    'TotalDose': totalDose,
                    'NbVolumes': nbVolumes,
                    'TotalSessions': totalSessions}
            
            for key in data.keys() :
                if data[key] == None :
                    data[key] = ''
            data = [data]
            prediction = requests.post(url=API_ENDPOINT, json=data).json()

    st.success(prediction if isinstance(prediction, str) or len(prediction) == 0 else f"Predicted time is {prediction[0]['time_in_months']} months")
    

if __name__ == "__main__":
    main()



