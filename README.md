# Forecasting LoRa RSSI

This repository contains the code for my master's dissertation on forecasting the RSSI (Received Signal Strength Indicator) metric in LoRa networks using a combination of machine learning algorithms and wavelet transforms. 

## Developed by:

### **Lorenzo Moreira Donatti**  
`Computer Engineer & MSc Student - UFSM`

📩 **Feel free to contact me:**  
✉️ [lorenzo.donatti@acad.ufsm.br](mailto:lorenzo.donatti@acad.ufsm.br)
## Overview

This repository leverages the data from the study [1] to apply multiple machine learning algorithms, combined with wavelet transforms, to forecast LoRa RSSI signals under various devices and conditions. The goal is to predict the RSSI values accurately, enabling better performance monitoring and optimization for LoRaWAN networks.

The following machine learning models and wavelet transform methods are used in the forecasting pipeline:
- **Machine Learning Models**: [SVMs, Random Forests, GradientBoosting, BiLSTM, CNN-BiLSTM.]
- **Wavelet Transforms**: [Not strictly defined yet, but Daubechies and Haar have been tested.]

## Requirements

Ensure that you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `pywt`
- `matplotlib`
- `TensorFLow`
- `XGboost`
- `Scipy`

You can install the required packages using `pip`:

```bash
pip install -r requirements.txt #Not developed yet
```

This will:
- Automatically execute the pipeline.
- Process the data, apply machine learning models and wavelet transforms.
- Save the results in two JSON files:
  - `results.json` (main forecasting results)
  - `results_wavelet.json` (wavelet-transformed data results)
- Generate images visualizing the forecasting results.

## Pipeline Details

The pipeline consists of the following steps:
1. **Data Preprocessing**: Raw RSSI data is cleaned and normalized for use.
2. **Wavelet Transform**: Wavelet decomposition is applied to the RSSI data to extract features for forecasting.
3. **Model Training and Forecasting**: Various machine learning algorithms are trained using the preprocessed data, and forecasts are generated.
4. **Results**: The model's forecasting performance is evaluated and saved.

## Under Development

This repository is still under active development. New features and models will be added in the future.

## References

[1] - Goldoni E., Savazzi P., Favalli L., Vizziello A. Correlation between weather and signal strength in LoRaWAN networks: An extensive dataset.
