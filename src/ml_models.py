from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, BatchNormalization


def build_model_CNN_bilstm(input_shape):
    model = Sequential()

    model.add(Conv1D(64, 2, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())

    # Camada convolucional adicional para maior captura de padrões locais
    model.add(Conv1D(128, 2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())

    # Camada convolucional adicional para maior captura de padrões locais
    model.add(Conv1D(256, 2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(256, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(16, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(8, activation='relu', return_sequences=False)))
    model.add(Dense(1)) 

    model.compile(optimizer='adam', loss='mse')

    return model


def build_model_bilstm(input_shape):
    model = Sequential()

    model.add(Bidirectional(LSTM(256, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(16, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(8, activation='relu', return_sequences=False)))
    model.add(Dense(1)) 

    model.compile(optimizer='adam', loss='mse')

    return model



def build_model_lstm(input_shape):
    model = Sequential()

    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(8, activation='relu', return_sequences=False))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model