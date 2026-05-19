#LOPEZ_JUAN-ProIA2907
# preprocess.py
import pandas as pd
import numpy as np
from config import LSTM_WINDOW_SIZE

def compute_delays(df):
    """Calcula delay_to_gate usando solo la hora del día (evita problemas de fechas)."""
    df = df.copy()
    # Convertir a datetime (si falla, coerce a NaT)
    df['slot'] = pd.to_datetime(df['asa_FechaHoraSlot'], errors='coerce')
    df['posicion2'] = pd.to_datetime(df['asa_FechaHoraPosicion2'], errors='coerce')
    
    # Extraer minutos desde medianoche
    slot_min = df['slot'].dt.hour * 60 + df['slot'].dt.minute
    pos_min = df['posicion2'].dt.hour * 60 + df['posicion2'].dt.minute
    
    # Diferencia horaria (puede ser negativa)
    delay = pos_min - slot_min
    # Ajuste por cruce de medianoche (valores entre -720 y +720 minutos)
    delay = np.where(delay < -720, delay + 1440, delay)
    delay = np.where(delay > 720, delay - 1440, delay)
    
    df['delay_to_gate'] = delay
    # Eliminar valores extremos (> 3 horas)
    df = df[df['delay_to_gate'].abs() < 180]
    return df

def add_time_features(df):
    df = df.copy()
    hour = df['slot'].dt.hour + df['slot'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(float)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(float)
    day = df['slot'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day / 7).astype(float)
    df['day_cos'] = np.cos(2 * np.pi * day / 7).astype(float)
    return df

def create_lagged_features(df, target='delay_to_gate', lags=LSTM_WINDOW_SIZE):
    df = df.copy()
    # Asegurar que el target sea numérico
    df[target] = pd.to_numeric(df[target], errors='coerce')
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df[target].shift(lag).astype(float)
    # Eliminar filas con NaN
    df = df.dropna()
    return df

def prepare_data(df, lags=LSTM_WINDOW_SIZE):
    """Pipeline completo: delays, features horarias, rezagos."""
    df = compute_delays(df)
    df = add_time_features(df)
    df = df.sort_values('slot')
    df = create_lagged_features(df, lags=lags)
    feature_cols = [f'lag_{i}' for i in range(1, lags+1)] + ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    X = df[feature_cols]
    y = df['delay_to_gate']
    return X, y, df, feature_cols