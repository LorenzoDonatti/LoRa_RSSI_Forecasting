from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

from src.ml_models import *
from src.proccessing_functions import *

import pandas as pd


def support_vector_machines(df_train, df_test):

    print("     --------------------------------")
    print("     Treinando Support Vector Machine")
    print("     --------------------------------")

    svm_model = SVR().fit(df_train.iloc[:,1:], 
                          df_train['RSSI'])
    
    y_pred = svm_model.predict(df_test.iloc[:,1:])
    
    return y_pred


def random_forest(df_train, df_test):

    print("     -----------------------")
    print("     Treinando Random Forest")
    print("     -----------------------")

    rf_model = RandomForestRegressor().fit(df_train.iloc[:,1:], 
                                           df_train['RSSI'])
    
    y_pred = rf_model.predict(df_test.iloc[:,1:])
    
    return y_pred


def xgboost(df_train, df_test):

    print("     -----------------")
    print("     Treinando XGBoost")
    print("     -----------------")

    xgb_model = XGBRegressor().fit(df_train.iloc[:,1:], 
                                   df_train['RSSI'])
    
    y_pred = xgb_model.predict(df_test.iloc[:,1:])

    return y_pred


def lstm(df_train, df_test, max_lag):

    print("     --------------")
    print("     Treinando LSTM")
    print("     --------------")

    X_train = df_train.iloc[:,1:].values.reshape((df_train.iloc[:,1:].shape[0], 
                                                  max_lag, 1))
    
    X_test = df_test.iloc[:,1:].values.reshape((df_test.iloc[:,1:].shape[0], 
                                                max_lag, 1))
    
    model = build_model_lstm((max_lag, 1))
    model.fit(X_train, df_train['RSSI'],
              epochs=64, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)

    return y_pred


def bilstm(df_train, df_test, max_lag):

    print("     ----------------")
    print("     Treinando BILSTM")
    print("     ----------------")

    X_train = df_train.iloc[:,1:].values.reshape((df_train.iloc[:,1:].shape[0], 
                                                  max_lag, 1))
    
    X_test = df_test.iloc[:,1:].values.reshape((df_test.iloc[:,1:].shape[0], 
                                                max_lag, 1))
    
    model = build_model_bilstm((max_lag, 1))
    model.fit(X_train, df_train['RSSI'],
              epochs=64, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)

    return y_pred

def CNN_bilstm(df_train, df_test, max_lag):

    print("     ----------------")
    print("     Treinando CNN-BILSTM")
    print("     ----------------")

    X_train = df_train.iloc[:,1:].values.reshape((df_train.iloc[:,1:].shape[0], 
                                                  max_lag, 1))
    
    X_test = df_test.iloc[:,1:].values.reshape((df_test.iloc[:,1:].shape[0], 
                                                max_lag, 1))
    
    model = build_model_CNN_bilstm((max_lag, 1))
    model.fit(X_train, df_train['RSSI'],
              epochs=64, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)

    return y_pred