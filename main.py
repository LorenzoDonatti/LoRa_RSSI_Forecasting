from src.ml_functions import *
from src.ml_models import *
from src.pipeline import *
from src.proccessing_functions import *
from src.modwt import *
from src.create_graphs import *

import numpy as np
import json

import matplotlib.pyplot as plt
plt.style.use('bmh')


def run_pipeline(outputs, algorithms, max_lag):

    final_metrics = {}
    final_metrics_wavelet = {}

    for i, output in enumerate(outputs):

        print("-----------------------------------------------------")
        print(f'INICIANDO O RSSI_{i+1}')
        print("-----------------------------------------------------")

        plot_original_data(output, i)

        output_train, output_test = split_train_test(output)

        models_output = run_ml(output_train, output_test, max_lag=max_lag)

        wave_type = 'db3'
        level = 2

        coeff, coeff_test = modwt(output_train['RSSI'], wave_type, level=level), modwt(output_test['RSSI'], wave_type, level=level)
        
        algorithm_dict = {algorithm: [] for algorithm in algorithms}

        for j in range(len(coeff)):

            print("         --------------------------------------------")
            print("         Inicializando Wavelet")
            print("         --------------------------------------------")

            y_preds = run_ml(pd.DataFrame({'RSSI': coeff[j]}), pd.DataFrame({'RSSI': coeff_test[j]}))

            [algorithm_dict[algorithm].append(y_preds[algorithm].flatten()) for algorithm in algorithms]


        models_output_wavelet = {algorithm: imodwt(np.array(algorithm_dict[algorithm]), wave_type) for algorithm in algorithms}

        final_metrics[f'RSSI_0{i+1}'] = metrics(output_test, models_output, max_lag=max_lag)
        final_metrics_wavelet[f'RSSI_0{i+1}'] = metrics(output_test, models_output_wavelet, max_lag=max_lag)

        print(final_metrics)
        print("--------------------------------------------")
        print(final_metrics_wavelet)

    return final_metrics, final_metrics_wavelet

algorithms = ['cnn_bilstm','bilstm', 'lstm', 'svm', 'rf', 'xgb']
url = "https://raw.githubusercontent.com/emanueleg/lora-rssi/master/vineyard-2021_data/combined_hourly_data.csv"

dfs = acquire_data(url)

data, data_wavelet = run_pipeline(dfs, algorithms, max_lag=8)

# Chamar a função para gerar e salvar os gráficos
plot_forecasting(data, algorithms=algorithms)
plot_error_metrics(data, algorithms=algorithms)

plot_forecasting(data, algorithms=algorithms, wavelet=True)
plot_error_metrics(data, algorithms=algorithms, wavelet=True)

with open("results.json", "w") as arquivo1:
    json.dump(data, arquivo1, indent=4)

with open("results_wavelet.json", "w") as arquivo2:
    json.dump(data_wavelet, arquivo2, indent=4)

print("\n----------------------------------------------------------")
print("Dicionários salvos em results.json e results_wavelet.json!")
print("----------------------------------------------------------\n")