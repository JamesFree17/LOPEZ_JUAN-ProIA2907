#LOPEZ_JUAN-ProIA2907
# train_model.py
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from db_connector import load_all_flights
from preprocess import prepare_data
from config import MODEL_PATH, FEATURE_COLS_PATH, LSTM_WINDOW_SIZE

def main():
    print("Cargando datos desde la base de datos...")
    df = load_all_flights()
    print(f"Total registros cargados: {len(df)}")
    
    # Preprocesar
    X, y, df_full, feature_cols = prepare_data(df, lags=LSTM_WINDOW_SIZE)
    print(f"Muestras disponibles después de preprocesar: {len(X)}")
    
    # División temporal (70% entrenamiento, 30% prueba)
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print("Entrenando LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE en prueba: {mae:.2f} minutos")
    
    # Guardar modelo y columnas
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    
    # Guardar los datos de simulación (los últimos 30% ordenados temporalmente)
    df_sim = df_full.iloc[split:].copy()
    df_sim.to_csv('datos_simulacion.csv', index=False)
    print(f"Datos de simulación guardados ({len(df_sim)} vuelos)")

if __name__ == "__main__":
    main()