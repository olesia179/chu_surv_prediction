import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from typing import cast, Optional
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder


class XGBModel:
    
    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
    ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # params = {'objective': 'survival:aft', 
    #           'eval_metric': 'aft-nloglik', 
    #           'tree_method': 'auto', 
    #           'seed': 42, 
    #           'verbosity': 0, 
    #           'reg_alpha': np.float64(1.885273386835653), 
    #           'reg_lambda': np.float64(1.3866093699752953), 
    #           'min_child_weight': np.int64(11), 
    #           'max_depth': np.int64(3), 
    #           'learning_rate': 0.001, 
    #           'subsample': np.float64(0.8526048126701572), 
    #           'colsample_bytree': np.float64(0.7348574730865784), 
    #           'aft_loss_distribution': 'logistic', 
    #           'aft_loss_distribution_scale': np.float64(0.5219336218198749)}
    # params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'auto', 'seed': 42, 'verbosity': 0, 'reg_alpha': np.float64(0.3906708939012361), 'reg_lambda': np.float64(2.654905091866789), 'min_child_weight': np.int64(25), 'max_depth': np.int64(4), 'learning_rate': 0.001, 'subsample': np.float64(0.9964285208619511), 'colsample_bytree': np.float64(0.8253269622787088), 'aft_loss_distribution': 'logistic', 'aft_loss_distribution_scale': np.float64(0.5966331314764576)}
    # params = {'objective': 'survival:aft', 'aft_loss_distribution': 'logistic', 'aft_loss_distribution_scale': 1.0}
    params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'auto', 'seed': 42, 'verbosity': 0, 'reg_alpha': np.float64(0.5052099414307786), 'reg_lambda': np.float64(0.6377107884457474), 'min_child_weight': np.int64(5), 'max_depth': np.int64(4), 'learning_rate': 0.001, 'subsample': np.float64(0.8669538641818011), 'colsample_bytree': np.float64(0.8951288066582845), 'aft_loss_distribution': 'logistic', 'aft_loss_distribution_scale': np.float64(0.9914290128527132)}
    
    @staticmethod
    def create_dmatrix(X: Optional[pd.DataFrame], y: Optional[np.ndarray]) -> xgb.DMatrix:
        dmat = xgb.DMatrix(X, label=y['time_in_months'])
        dmat.set_float_info('label_lower_bound', y['time_in_months'])
        dmat.set_float_info('label_upper_bound', np.where(y['dead']==1, y['time_in_months'], np.inf))
        return dmat

    def prepare_arrays(self, df: pd.DataFrame, num_features: list[str], cat_ohe_features: list[str], cat_ord_features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        
        # cat_ohe_features = list(filter(lambda x : x.rsplit('_', 1)[0] in cat_ohe_features, df.columns.to_list()))
        features = num_features + cat_ohe_features + cat_ord_features

        X = df[features]
        
        df['time_in_months'] = df['time_in_months'].fillna(999)
        y = Surv.from_dataframe('dead', 'time_in_months', df)
        return X, y

    def split_data(self, df: pd.DataFrame, num_features: list[str], cat_ohe_features: list[str], cat_ord_features: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:

        X, _ = self.prepare_arrays(df, num_features, cat_ohe_features, cat_ord_features)

        all_indices = df.index.to_numpy()
        train_indices, test_indices = train_test_split(
            all_indices, 
            test_size=0.2, 
            random_state=42
        )
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train = Surv.from_dataframe('dead', 'time_in_months', df.loc[train_indices])
        y_test = Surv.from_dataframe('dead', 'time_in_months', df.loc[test_indices])
        return X_train, y_train, train_indices, X_test, y_test, test_indices

    def train_model(self, df: pd.DataFrame, num_features: list[str], cat_ohe_features: list[str], cat_ord_features: list[str], full_ds=False) -> tuple[xgb.Booster, Optional[xgb.DMatrix], Optional[np.ndarray]]:
        X_train, y_train, train_indices, X_test, y_test, test_indices = \
            self.split_data(df, num_features, cat_ohe_features, cat_ord_features)
        preprocessor = ColumnTransformer(
            [
                ("numerical", StandardScaler(), num_features),
                ("cat_ord", self.ord_encoder, cat_ord_features),
                ("cat_ohe", self.ohe_encoder, cat_ohe_features)
            ],
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")
        X_train = cast(pd.DataFrame, preprocessor.fit_transform(X_train))
        
        dtrain = XGBModel.create_dmatrix(X_train, y_train)
        train_kwargs = {
            'params': self.params,
            'dtrain': dtrain,
            'verbose_eval': 500
        }
        
        X_test = cast(pd.DataFrame, preprocessor.transform(X_test))
        dtest = XGBModel.create_dmatrix(X_test, y_test)
        train_kwargs['num_boost_round'] = 30000
        train_kwargs['evals'] = [(dtest, 'test')]
        train_kwargs['early_stopping_rounds'] = 500
        
        model = xgb.train(**train_kwargs)

        if full_ds:
            dtest = None
            test_indices = None
            X, y = self.prepare_arrays(df, num_features, cat_ohe_features, cat_ord_features)
            X = cast(pd.DataFrame, preprocessor.fit_transform(X))
            joblib.dump(preprocessor, './data/preprocessor.pkl')

            dtrain = self.create_dmatrix(X, y)
            train_kwargs['num_boost_round'] = model.best_iteration + 1
            train_kwargs.pop('evals')
            train_kwargs.pop('early_stopping_rounds')

            model = xgb.train(**train_kwargs)
            model.save_model('./data/model.ubj')

        return model, dtest, test_indices

    def predict_train(self, df: pd.DataFrame, num_features: list[str], cat_ohe_features: list[str], cat_ord_features: list[str]) -> pd.DataFrame:

        model, dtest, test_indices = self.train_model(df, num_features, cat_ohe_features, cat_ord_features)

        if (dmatrix := dtest) is not None and (test_idxs := test_indices) is not None:

            times_preds = model.predict(dmatrix, iteration_range=(0, model.best_iteration + 1))

            time_test = df.loc[test_idxs, 'time_in_months']
            event_test = df.loc[test_idxs, 'dead']
            print(f"XGBoost c-index: {concordance_index(time_test, times_preds, event_test)}")

            times_preds = np.round(times_preds).astype(int)

            result = pd.DataFrame({'xgb_aft_time': times_preds, 'true_time' : time_test, 
                                   'ES_VIM': df.loc[test_idxs, 'ES_VIM'], 'dead': event_test})
            result = result[result['dead'] == 1].reset_index(drop=True)
            print(result[['xgb_aft_time', 'true_time']].describe())
            print(result[['xgb_aft_time', 'true_time']].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
            return result
        else:
            return pd.DataFrame()

    def predict_full(self, df_full: pd.DataFrame, num_features: list[str], cat_ohe_features: list[str], cat_ord_features: list[str], df_val: pd.DataFrame) -> pd.DataFrame:

        model, _, _ = self.train_model(df_full, num_features, cat_ohe_features, cat_ord_features, True)

        X, y = self.prepare_arrays(df_val, num_features, cat_ohe_features, cat_ord_features)
        preprocessor = joblib.load('./data/preprocessor.pkl')
        X = cast(pd.DataFrame, preprocessor.transform(X))
        dtest = XGBModel.create_dmatrix(X, y)
        
        times_preds = model.predict(dtest)

        time_test = df_val['time_in_months']
        event_test = df_val['dead']
        print(f"XGBoost c-index: {concordance_index(time_test, times_preds, event_test)}")

        times_preds = np.round(times_preds).astype(int)

        result = pd.DataFrame({'xgb_aft_time': times_preds, 'true_time' : time_test, 
                                'ES_VIM': df_val['ES_VIM'], 'dead': event_test})
        result = result[result['dead'] == 1].reset_index(drop=True)
        print(result[['xgb_aft_time', 'true_time']].describe())
        print(result[['xgb_aft_time', 'true_time']].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
        return result
