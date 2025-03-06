import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def metrics(df_test, dict, max_lag=8):

    results = {}

    results["test_values"] = df_test['RSSI'].iloc[max_lag:].values.tolist()
    
    for algorithm, y_pred in dict.items():
        mae = mean_absolute_error(df_test['RSSI'].iloc[max_lag:], y_pred.flatten())
        rmse = mean_squared_error(df_test['RSSI'].iloc[max_lag:], y_pred.flatten(), squared=False)
        mape = mean_absolute_percentage_error(df_test['RSSI'].iloc[max_lag:], y_pred.flatten())
        
        results[f"predicted_values_{algorithm}"] = y_pred.flatten().tolist()
        results[f"mae_{algorithm}"] = mae
        results[f"rmse_{algorithm}"] = rmse
        results[f"mape_{algorithm}"] = mape
    
    return results


def create_lags(df, column_name, max_lag=8):

    for lag in range(1, max_lag + 1):
        df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)

    # Remove as linhas com NaNs (que surgem devido ao shift)
    df_cleaned = df.dropna(subset=[f'{column_name}_lag_{lag}' for lag in range(1, max_lag + 1)])

    return df_cleaned


def acquire_data(url):

    df = pd.read_csv(url, sep=';')

    # Selecione as colunas de RSSI
    rssi_columns = ['RSSI_01', 'RSSI_02', 'RSSI_03', 'RSSI_04', 'RSSI_05', 'RSSI_06', 'RSSI_07', 'RSSI_08']

    # Crie um dicionário para armazenar os DataFrames
    rssi_dfs = []

    # Para cada coluna de RSSI, crie um DataFrame correspondente
    for col in rssi_columns:
        df[col].fillna(df[col].mean(), inplace=True)
        rssi_dfs.append(df[[col]].copy())

    return rssi_dfs


def split_train_test(data, train_ratio=0.8):

    data.rename(columns={col: 'RSSI' for col in data.filter(like='RSSI').columns}, inplace=True)

    n = int(len(data) * train_ratio)
    
    datatrain = data.iloc[:n].copy()
    datatest = data.iloc[n:].copy()
    
    return datatrain, datatest