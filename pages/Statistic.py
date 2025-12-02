import streamlit as st
from modules.loader.DataLoader import DataLoader
from modules.processor.DataProcessor import DataProcessor
from modules.describer.DataDescriber import DataDescriber as describer
import pandas as pd
from typing import Optional

st.set_page_config(page_title="Dataset statistic", page_icon="📊")

st.markdown("# Dataset overview")
st.sidebar.header("Dataset statistic")

@st.cache_data
def load_data() -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    data_loader = DataLoader()
    df_raw = data_loader.load_from_csv()
    data_processor = DataProcessor()
    df, df_filtered_out = data_processor.get_df(df_raw)
    return df, df_filtered_out

# Description of the sample
s_num_fields = ['age', 'BMI', 'DIA_MAX_TUMEUR_1', 'DIA_MAX_TUMEUR_2', 'DIA_MAX_TUMEUR_3', 'DIA_MAX_TUMEUR_4']
s_cat_fields = ['gender', 'FUMEUROUINON', 'ICD9', 'dead', 'PACEMAKER_ON', 'DIABETIQUE_ON', 'ALLERGIE_ON', 'INSULO_ON', 'INSUFFURENALE_ON']

# Description of the SRBT intents and RT plans
p_num_fields = ['TotalDose', 'NbVolumes', 'TotalSessions']
p_cat_fields = ['DEGRURG', 'CHIMOANT_ON', 'NRSF11', 'OMS', 'MOBPAT', 'IRRA_ANT_ON', 'IRRANT_OLOCA', 'CHIRURGIE_ANT_ON']

# The Coms
c_cat_fields = ['T', 'N', 'M']

df, df_filtered_out = load_data()

st.markdown(f'## Filtered out samples count: {df_filtered_out.shape[0]}')

st.markdown('## Sample description:')
st.write(describer.describe_sample(df, s_num_fields, s_cat_fields))
st.markdown('## SRBT intents and RT plans description:')
st.write(describer.describe_sample(df, p_num_fields, p_cat_fields))
st.markdown('## TNM staging distribution:')
st.write(describer.describe_sample(df, cat_fields = c_cat_fields))
st.markdown('## Outcomes description:')
st.write(describer.describe_outcomes(df))
st.markdown('## Estimations statistics:')
st.write(describer.estimations_stats(df))

