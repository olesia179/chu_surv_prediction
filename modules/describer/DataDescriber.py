import pandas as pd
import streamlit as st
class DataDescriber:

    @staticmethod
    def describe_sample(df: pd.DataFrame, num_fields: list[str] = [], cat_fields: list[str] = []):
        if len(num_fields) > 0:
            st.write("Numerical fields:")
            st.write(df[num_fields].describe())
        if len(cat_fields) > 0:
            st.write("\nCategorical fields:")
            for cat_field in cat_fields:
                st.write(df[cat_field].value_counts())

    @staticmethod
    def describe_outcomes(df: pd.DataFrame):
        # st.write(df['dead_in_months'].value_counts())
        # st.write(df['alive_in_months'].value_counts())

        alive_after_0_months = (df['dead_in_months']  > 0).sum() + df['alive_in_months'].notna().sum()
        alive_after_1_months = (df['dead_in_months']  > 1).sum() + df['alive_in_months'].notna().sum()
        alive_after_2_months = (df['dead_in_months']  > 2).sum() + df['alive_in_months'].notna().sum()
        alive_after_3_months = (df['dead_in_months']  > 3).sum() + df['alive_in_months'].notna().sum()
        alive_after_4_months = (df['dead_in_months']  > 4).sum() + df['alive_in_months'].notna().sum()
        alive_after_5_months = (df['dead_in_months']  > 5).sum() + df['alive_in_months'].notna().sum()
        alive_after_6_months = (df['dead_in_months']  > 6).sum() + df['alive_in_months'].notna().sum()

        total_len = df.shape[0]

        result = pd.DataFrame({'Alive' : ['same month', 'after 1 months', 'after 2 months',
                                  'after 3 months', 'after 4 months', 'after 5 months', 'after 6 months'],
                               'Count' : [alive_after_0_months, alive_after_1_months, alive_after_2_months,
                                          alive_after_3_months, alive_after_4_months, alive_after_5_months, alive_after_6_months],
                               'Percentage' : [alive_after_0_months / total_len * 100, alive_after_1_months / total_len * 100,
                                               alive_after_2_months / total_len * 100, alive_after_3_months / total_len * 100,
                                               alive_after_4_months / total_len * 100, alive_after_5_months / total_len * 100,
                                               alive_after_6_months / total_len * 100]})
        st.write(result)


    @staticmethod
    def estimations_stats(df: pd.DataFrame):
        estimated_idx = df['ES_VIM'].notna()
        dead_idx = df['dead_in_months'].notna()
        alive_idx = ~dead_idx

        df_with_estimated_death = df[estimated_idx & dead_idx]

        df_alive = df[estimated_idx & alive_idx].copy()

        total_est_len = df_with_estimated_death.shape[0] + df_alive.shape[0]
        
        propre_idx = (df_with_estimated_death['dead_in_months'] >= df_with_estimated_death['ES_VIM'] - 1) & (
            df_with_estimated_death['dead_in_months'] <= df_with_estimated_death['ES_VIM'] + 1)
        over_idx = (df_with_estimated_death['dead_in_months'] < df_with_estimated_death['ES_VIM'] - 1)
        under_idx = (df_with_estimated_death['dead_in_months'] > df_with_estimated_death['ES_VIM'] + 1)
        alive_under_idx = (df_alive['alive_in_months'] >= df_alive['ES_VIM'] + 1)

        dead_under_cnt = under_idx.sum()
        alive_under_cnt = alive_under_idx.sum()
        result = pd.DataFrame({'Estimation' : ['Right estimations', 'Overestimated', 'Underestimated'],
                               'Count' : [f"{propre_idx.sum()}", f"{over_idx.sum()}", 
                                          f"{dead_under_cnt + alive_under_cnt}: {dead_under_cnt} dead + {alive_under_cnt} alive"],
                               'Percentage' : [propre_idx.sum() / total_est_len * 100,
                                               over_idx.sum() / total_est_len * 100,
                                               (dead_under_cnt + alive_under_cnt) / total_est_len * 100]})
        # f'{(dead_under_cnt + alive_under_cnt) / total_est_len * 100}: {dead_under_cnt / total_est_len * 100} dead + {alive_under_cnt / total_est_len * 100} alive'
        st.write(result)

