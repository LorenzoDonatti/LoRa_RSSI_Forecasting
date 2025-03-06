from src.proccessing_functions import create_lags
from src.ml_functions import *

import pandas as pd

def run_ml(df_train, df_test, max_lag=8):

    results = []

    df_train = create_lags(df_train, 'RSSI', max_lag)
    df_test = create_lags(df_test, 'RSSI', max_lag)

    return {"cnn_bilstm": CNN_bilstm(df_train, df_test, max_lag),
            "bilstm": bilstm(df_train, df_test, max_lag),
            "lstm": lstm(df_train, df_test, max_lag),
            "svm": support_vector_machines(df_train, df_test),
            "rf": random_forest(df_train, df_test),
            "xgb": xgboost(df_train, df_test)
    }