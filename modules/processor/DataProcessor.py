import pandas as pd
from typing import Optional

class DataProcessor:

    def __init__(self, num_features: Optional[list[str]] = None, cat_ohe_features: Optional[list[str]] = None, cat_ord_features: Optional[list[str]] = None) -> None:
        self.num_features = num_features
        self.cat_ohe_features = cat_ohe_features
        self.cat_ord_features = cat_ord_features

    columns_to_exclude = ['IntentId', # id
                            # 'numnat', # numero national
                            'startDate', 
                            'hcpId', # docteurId
                            'BTNDATE_RXTHDATE1',
                            'BTNDATE_CHIRANT',
                            'BTNDATE_CHIMIOANT',
                            'BTNDATE_ CHIMIOANT_FIN',
                            'LOCPAT',
                            'CAPPOSTABLE',
                            'FUMEUR_LISTE',
                            'MALMALON',
                            'BUTRAD1',
                            'TYPE',
                            'RYTHCHCC',
                            'LOCALISATION_VOL_1',
                            'DOSE_TOT_GR_1',
                            'LOC_VERT_1',
                            'DENSIT_TRAIT_1',
                            'TRAITLOCALI_1', # empty
                            'LOCALISATION_VOL_2',
                            'DOSE_TOT_GR_2',
                            'LOC_VERT_2',
                            'DENSIT_TRAIT_2',
                            'TRAITLOCALI_2',
                            'LOCALISATION_VOL_3',
                            'DOSE_TOT_GR_3',
                            'LOC_VERT_3',
                            'DENSIT_TRAIT_3',
                            'TRAITLOCALI_3',
                            'LOCALISATION_VOL_4',
                            'DOSE_TOT_GR_4',
                            'LOC_VERT_4',
                            'DENSIT_TRAIT_4',
                            'TRAITLOCALI_4',
                            'BirthDate',
                            'birthYear',
                            'DeathDate',
                            'socialDate',
                            'VITSEUL',
                            'ETATMATR',
                            'MAISONET',
                            'APPARTEMENT',
                            'PATCONDUIT',
                            'PATIENTSOCPROF',
                            'LESAIDANTS',
                            'AIDANTSPROF',
                            'AIDANTAIDEON',
                            'BodySite', 'CodeIcd', 'Morpho', 'Orclat', 'DateIncidence', 'TreatmentCodes', 'TumorBehavior',
                            'ECOG', 'OrcDiffer', 'TreatmentGoal', 'NbCancer', 'deathYear', 'deathMonth']
    
    def fix_types(self, df):
        
        # df['numnat'] = df['numnat'].astype('string')
    
        df['IRRA_ANT_ON'] = df['IRRA_ANT_ON'].fillna(0).astype(int)
        df['IRRANT_OLOCA'] = df['IRRANT_OLOCA'].astype('Int64')

        df['BMI'] = df['BMI'].fillna(0).astype(float)
        # df['PDS'] = df['PDS'].fillna(0).astype(float)
        # df['TAILLE'] = df['TAILLE'].fillna(0).astype(float)

        df['NRSF11'] = df['NRSF11'].astype('Int64') # .fillna(0)
        df['OMS'] = df['OMS'].astype('Int64') # .fillna(0)
        # df['ES_VIM'] = df['ES_VIM'].astype('Int64')

        df['MOBPAT'] = df['MOBPAT'].astype('string') # .fillna('VAL')
        df['INSULO_ON'] = df['INSULO_ON'].fillna(0).astype(int)
        # df['METFORMINE'] = df['METFORMINE'].fillna(0).astype(int)

        df['DIA_MAX_TUMEUR_1'] = df['DIA_MAX_TUMEUR_1'].fillna(0).astype(float)
        # df['NBR_FRAC_1'] = df['NBR_FRAC_1'].astype('Int64')
        # df['DOSE_TOT_GR_1'] = df['DOSE_TOT_GR_1'].fillna(0).astype(float)
        df['DIA_MAX_TUMEUR_2'] = df['DIA_MAX_TUMEUR_2'].fillna(0).astype(float)
        # df['NBR_FRAC_2'] = df['NBR_FRAC_2'].astype('Int64')
        # df['DOSE_TOT_GR_2'] = df['DOSE_TOT_GR_2'].fillna(0).astype(float)
        df['DIA_MAX_TUMEUR_3'] = df['DIA_MAX_TUMEUR_3'].fillna(0).astype(float)
        # df['NBR_FRAC_3'] = df['NBR_FRAC_3'].astype('Int64')
        # df['DOSE_TOT_GR_3'] = df['DOSE_TOT_GR_3'].fillna(0).astype(float)
        df['DIA_MAX_TUMEUR_4'] = df['DIA_MAX_TUMEUR_4'].fillna(0).astype(float)
        # df['NBR_FRAC_4'] = df['NBR_FRAC_4'].astype('Int64')
        # df['DOSE_TOT_GR_4'] = df['DOSE_TOT_GR_4'].fillna(0).astype(float)

        df['gender'] = df['gender'].astype('category').cat.codes

        # df['dose_1'] = df['DOSE_TOT_GR_1'].where(df['DOSE_TOT_GR_1'] >= 1000, df['DOSE_TOT_GR_1'] * 100).astype(int)
        # df['dose_2'] = df['DOSE_TOT_GR_2'].where(df['DOSE_TOT_GR_2'] >= 1000, df['DOSE_TOT_GR_2'] * 100).astype(int)
        # df['dose_3'] = df['DOSE_TOT_GR_3'].where(df['DOSE_TOT_GR_3'] >= 1000, df['DOSE_TOT_GR_3'] * 100).astype(int)
        # df['dose_4'] = df['DOSE_TOT_GR_4'].where(df['DOSE_TOT_GR_4'] >= 1000, df['DOSE_TOT_GR_4'] * 100).astype(int)
        
        df['TotalDose'] = df['TotalDose'].astype('Int64')
        df['NbVolumes'] = df['NbVolumes'].astype('Int64')
        df['TotalSessions'] = df['TotalSessions'].astype('Int64')

        # df['CT'] = df['CT'].astype('string')
        # df['CN'] = df['CN'].astype('string')
        # df['CM'] = df['CM'].astype('string')
        # df['PT'] = df['PT'].astype('string')
        # df['PN'] = df['PN'].astype('string')
        # df['PM'] = df['PM'].astype('string')
        df['T'] = df['T'].astype('string')
        df['N'] = df['N'].astype('string')
        df['M'] = df['M'].astype('string')

        df['DEGRURG'] = df['DEGRURG'].replace({
            2: 1,
            4: 0
        }).astype(int)

        return df
    
    def create_new_fields(self, df):

        if 'startDate' in df.columns:
            df['startYear'] = df['startDate'].str.slice(0,4).astype(int)
            df['startMonth'] = df['startDate'].str.slice(5,7).astype(int)

        if 'DeathDate' in df.columns:
            df['deathYear'] = pd.to_numeric(df['DeathDate'].str.slice(0,4), errors='coerce').astype('Int64')
            df['deathMonth'] = pd.to_numeric(df['DeathDate'].str.slice(5,7), errors='coerce').astype('Int64')

        if 'BirthDate' in df.columns:
            df['birthYear'] = df['BirthDate'].str.slice(0,4).astype(int)
            df['age'] = (df['startYear'] - df['birthYear']).astype(int)

        return df
    
    def get_eligible_rows(self, df, last_obs_month = 9, last_obs_year = 2023):
        if 'dead' not in df.columns:
            return df, None
        eligible_idx = (df['startYear'] < last_obs_year) | ((df['startYear'] == last_obs_year) & (df['startMonth'] < last_obs_month - 6))
        return df[eligible_idx].copy(), df[~eligible_idx].copy()
    
    def create_dead_in_months_field(self, df):
        if 'dead' not in df.columns:
            return df
            
        dead_idx = (df['dead'] == 1)
        if dead_idx.sum() > 0:
            df.loc[dead_idx, 'time_in_months'] = ((df['deathYear'] - df['startYear']) * 12 + (df['deathMonth'] - df['startMonth'])).astype('Int64')
            df.loc[dead_idx, 'dead_in_months'] = df.loc[dead_idx, 'time_in_months']
        
        return df

    def create_alive_in_months_field(self, df, last_obs_month = 9, last_obs_year = 2023):

        if 'dead' not in df.columns:
            return df, pd.Series([False] * df.shape[0], index=df.index)
        
        dead_idx = (df['dead'] == 1)

        dead_after_idx = dead_idx & (
                (df['deathYear'] > last_obs_year) |
                ((df['deathYear'] == last_obs_year) &
                (df['deathMonth'].notna()) &
                (df['deathMonth'] >= last_obs_month))
            )
        # print(f"Number of excluded deaths: {dead_after_idx.sum()}")
        # estimated_idx = df['ES_VIM'].notna()
        # print(f"With ES_VIM: {(dead_after_idx & estimated_idx).sum()}")
        
        alive_idx = ~dead_idx
        # df_eligible_dead_after = df[dead_after_idx].copy()
        df.loc[alive_idx | dead_after_idx, 'time_in_months'] = ((last_obs_year - df['startYear']) * 12 + (last_obs_month - 1 - df['startMonth'])).astype('Int64')
        df.loc[alive_idx | dead_after_idx, 'alive_in_months'] = df.loc[alive_idx | dead_after_idx, 'time_in_months']
        
        return df, dead_after_idx
    
    def change_dead_status(self, df, dead_after_idx):
        if dead_after_idx.sum() > 0:
            df.loc[dead_after_idx, 'dead'] = 0
            df.loc[dead_after_idx, 'dead_in_months'] = None
        return df

    def impute_missing_values(self, df):
        df['IRRANT_OLOCA'] = df['IRRANT_OLOCA'].fillna(-1)
        df['NRSF11'] = df['NRSF11'].fillna(-1)
        df['OMS'] = df['OMS'].fillna(-1)
        df['MOBPAT'] = df['MOBPAT'].fillna('Unknown')
        df['IRRANT_OLOCA'] = df['IRRANT_OLOCA'].fillna(-1)
        df['TotalDose'] = df['TotalDose'].fillna(-1)
        df['NbVolumes'] = df['NbVolumes'].fillna(-1)
        df['TotalSessions'] = df['TotalSessions'].fillna(-1)
        df['T'] = df['T'].fillna('Unknown')#.astype('category').cat.codes
        df['N'] = df['N'].fillna('Unknown')#.astype('category').cat.codes
        df['M'] = df['M'].fillna('Unknown')#.astype('category').cat.codes
        return df

    def get_df(self, df_raw, last_obs_month = 9, last_obs_year = 2023) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        df = df_raw.copy()

        df = self.fix_types(df)
        df = self.create_new_fields(df)

        df = self.create_dead_in_months_field(df)
        df = self.impute_missing_values(df)

        df_filtered, df_not_filtered = self.get_eligible_rows(df, last_obs_month, last_obs_year)

        df, dead_after_idx = self.create_alive_in_months_field(df_filtered, last_obs_month, last_obs_year)

        df = self.change_dead_status(df, dead_after_idx)
        
        df = df.drop(columns=self.columns_to_exclude, errors='ignore')

        if (not_filtered := df_not_filtered) is None:
            return df, None
        
        return df.sort_values(by=['startYear', 'startMonth']), not_filtered.sort_values(by=['startYear', 'startMonth'])
