import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('classic')
plt.style.use('fivethirtyeight')

def plot_forecasting(data, algorithms=[], wavelet=False):
    for rssi_key, rssi_data in data.items():
        plt.figure(figsize=(10, 6))

        test_values = rssi_data['test_values']

        plt.plot(test_values, label=f'Real Values')
        
        for algorithm in algorithms:
          predicted_values = rssi_data[f'predicted_values_{algorithm}']
          
          plt.plot(predicted_values, label=f'Predicted Values - {algorithm}')
          
        # Adicionar título e rótulos
        plt.title(f'Comparison of Test and Predicted RSSI Values {rssi_key}', fontsize=14)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('RSSI Value', fontsize=12)
        plt.legend(loc='best', frameon=True, fontsize=11)
        plt.grid(False)
          
        if(wavelet==True):
            plt.savefig(f'data/images/forecasts/{rssi_key}_forecast_wavelet.png')
        else:
            plt.savefig(f'data/images/forecasts/{rssi_key}_forecast.png')
        plt.close()


def plot_error_metrics(data, algorithms=[], wavelet=False):
    for rssi_key, rssi_data in data.items():
        plt.figure(figsize=(10, 6))
        metrics = ['mae', 'rmse', 'mape']
        x = np.arange(len(metrics))

        bar_width = 0.12 
        offset = (len(algorithms) - 1) * bar_width / 2 

        for i, algorithm in enumerate(algorithms):
            values = [rssi_data[f'{metric}_{algorithm}'] for metric in metrics]
            plt.bar(x - offset + i * bar_width, values, bar_width, label=f'{algorithm}')

        plt.xticks(x, metrics, fontsize=12)
        plt.title(f'Error Comparison for {rssi_key}', fontsize=14)
        plt.ylabel('Error Value', fontsize=12)
        plt.legend(fontsize=12, loc='best')
        plt.grid(False)

        if (wavelet==True):
            plt.savefig(f'data/images/error_metrics/{rssi_key}_errors_wavelet.png')
        else:
            plt.savefig(f'data/images/error_metrics/{rssi_key}_errors.png')
        plt.close()


def plot_original_data(data, i):
        
        plt.figure(figsize=(12, 8))

        plt.plot(data.iloc[:,0].index, data.iloc[:,0].values, color='gray')

        # Adicionar título e rótulos
        plt.title(f'RSSI_{i} Values', fontsize=14)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('RSSI Value', fontsize=12)
        plt.grid(False)

        # Salvar o gráfico como imagem PNG
        plt.savefig(f'data/images/original_series/original_RSSI_{i}.png')
        plt.close()  # Fecha a figura para liberar memória