#LOPEZ_JUAN-ProIA2907
# simulador.py
import pandas as pd
import time
import joblib
import numpy as np
from datetime import datetime, timedelta
from db_connector import get_historical_data_before
from preprocess import compute_delays, add_time_features, create_lagged_features
from gate_optimizer import find_alternative_gate
from config import MODEL_PATH, FEATURE_COLS_PATH, LSTM_WINDOW_SIZE, TOLERANCE_MINUTES

def run_simulation(speed_factor=1.0, progress_callback=None):
    try:
        import os
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encuentra {MODEL_PATH}. Ejecuta train_model.py.")
        if not os.path.exists('datos_simulacion.csv'):
            raise FileNotFoundError("No se encuentra 'datos_simulacion.csv'. Ejecuta train_model.py.")
        
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        
        df_sim = pd.read_csv('datos_simulacion.csv', parse_dates=['slot'])
        if 'posicion2' not in df_sim.columns:
            if 'delay_to_gate' in df_sim.columns:
                df_sim['posicion2'] = pd.to_datetime(df_sim['slot']) + pd.to_timedelta(df_sim['delay_to_gate'], unit='m')
            else:
                raise KeyError("Falta 'posicion2' o 'delay_to_gate'")
        else:
            df_sim['posicion2'] = pd.to_datetime(df_sim['posicion2'])
        
        rename_map = {}
        if 'asa_Posicion' in df_sim.columns and 'puerta' not in df_sim.columns:
            rename_map['asa_Posicion'] = 'puerta'
        if 'asa_AerolineaICAO' in df_sim.columns and 'aerolinea' not in df_sim.columns:
            rename_map['asa_AerolineaICAO'] = 'aerolinea'
        if 'asa_NoVuelo' in df_sim.columns and 'vuelo' not in df_sim.columns:
            rename_map['asa_NoVuelo'] = 'vuelo'
        if 'asa_Terminal' in df_sim.columns and 'terminal' not in df_sim.columns:
            rename_map['asa_Terminal'] = 'terminal'
        df_sim.rename(columns=rename_map, inplace=True)
        
        df_sim['slot'] = pd.to_datetime(df_sim['slot'])
        df_sim['posicion2'] = pd.to_datetime(df_sim['posicion2'])
        df_sim = df_sim.sort_values('slot').reset_index(drop=True)
        
        first_slot = df_sim.iloc[0]['slot']
        df_hist = get_historical_data_before(first_slot)
        df_hist = compute_delays(df_hist)
        df_hist = add_time_features(df_hist)
        df_hist.rename(columns={
            'asa_Posicion': 'puerta',
            'asa_AerolineaICAO': 'aerolinea',
            'asa_NoVuelo': 'vuelo',
            'asa_Terminal': 'terminal'
        }, inplace=True)
        df_hist['slot'] = pd.to_datetime(df_hist['slot'])
        df_hist['posicion2'] = pd.to_datetime(df_hist['posicion2'])
        df_hist = df_hist.sort_values('slot')
        
        results = []
        total = len(df_sim)
        if progress_callback:
            progress_callback(0, total, "Iniciando simulación...")
        
        for i, vuelo in df_sim.iterrows():
            # Simulación SIN SLEEP (para pruebas rápidas)
            # time.sleep(...) se omite intencionalmente
            historico_hasta_antes = df_hist[df_hist['slot'] < vuelo['slot']].copy()
            if i > 0:
                simulados_pasados = df_sim.iloc[:i].copy()
                historico_hasta_antes = pd.concat([historico_hasta_antes, simulados_pasados], ignore_index=True)
            historico_hasta_antes = historico_hasta_antes.sort_values('slot')
            
            if len(historico_hasta_antes) < LSTM_WINDOW_SIZE:
                continue
            
            df_lag = create_lagged_features(historico_hasta_antes, lags=LSTM_WINDOW_SIZE)
            if df_lag.empty:
                continue
            
            last_row = df_lag.iloc[-1]
            X_pred = last_row[feature_cols].astype(float).fillna(0).values.reshape(1, -1)
            delay_pred = model.predict(X_pred)[0]
            
            start_time = vuelo['slot']
            end_time = start_time + timedelta(minutes=delay_pred + TOLERANCE_MINUTES)
            vuelos_en_ventana = historico_hasta_antes[
                (historico_hasta_antes['slot'] < end_time) & 
                (historico_hasta_antes['posicion2'] > start_time)
            ]
            ocupacion = None
            if not vuelos_en_ventana.empty:
                ocupacion = vuelos_en_ventana[['puerta', 'slot', 'posicion2']].rename(
                    columns={'slot': 'slot_inicio', 'posicion2': 'slot_fin'}
                )
            
            vuelo_dict = {
                'slot': vuelo['slot'],
                'aerolinea': vuelo['aerolinea'],
                'puerta_original': vuelo['puerta'],
                'terminal': str(vuelo['terminal'])
            }
            nueva_puerta = None
            if delay_pred > TOLERANCE_MINUTES:
                nueva_puerta = find_alternative_gate(vuelo_dict, delay_pred, ocupacion)
            
            results.append({
                'slot': vuelo['slot'],
                'aerolinea': vuelo['aerolinea'],
                'vuelo': vuelo['vuelo'],
                'puerta_original': vuelo['puerta'],
                'delay_predicho': round(delay_pred, 1),
                'puerta_sugerida': nueva_puerta if nueva_puerta else '-',
                'timestamp_simulacion': datetime.now()
            })
            
            if (i+1) % 50 == 0 or i+1 == total:
                msg = f"Procesados {i+1}/{total} vuelos"
                if progress_callback:
                    progress_callback(i+1, total, msg)
                else:
                    print(msg)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv('resultados_simulacion.csv', index=False)
        return df_results
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise Exception(error_msg)

if __name__ == "__main__":
    run_simulation(speed_factor=1)  # sin sleep real