import streamlit as st
from modules.loader.DataLoader import DataLoader
from modules.processor.DataProcessor import DataProcessor
from modules.model.XGBModel import XGBModel
import pandas as pd
from typing import Optional

st.set_page_config(page_title="Training", page_icon="📈")

st.markdown("# Survival Prediction")
st.sidebar.header("Training model")

num_features = ['age', 'gender', 'BMI', 'DIA_MAX_TUMEUR_1', 'DIA_MAX_TUMEUR_2', 'DIA_MAX_TUMEUR_3', 'DIA_MAX_TUMEUR_4', 
              'INSUFFURENALE_ON', 'FUMEUROUINON', 'DIABETIQUE_ON', 'INSULO_ON', 'PACEMAKER_ON', 'ALLERGIE_ON', 
              'TotalDose', 'NbVolumes', 'TotalSessions', 'DEGRURG', 'CHIMOANT_ON', 'IRRA_ANT_ON', 'CHIRURGIE_ANT_ON']

cat_ohe_features = ['ICD9', 'IRRANT_OLOCA', 'MOBPAT', 'T', 'N', 'M']

cat_ord_features = ['NRSF11', 'OMS']

@st.cache_data
def init(num, ohe, ord) -> tuple[pd.DataFrame, Optional[pd.DataFrame], XGBModel]:
    model = XGBModel()
    data_loader = DataLoader()
    df_raw = data_loader.load_from_csv()
    data_processor = DataProcessor(num, ohe, ord)
    df, df_filtered_out = data_processor.get_df(df_raw)
    return df, df_filtered_out, model

df, df_filtered_out, model = init(num_features, cat_ohe_features, cat_ord_features)

st.write(model.predict_train(df, num_features, cat_ohe_features, cat_ord_features))
if (df_filtered := df_filtered_out) is not None :
    st.write(model.predict_full(df, num_features, cat_ohe_features, cat_ord_features, df_filtered))