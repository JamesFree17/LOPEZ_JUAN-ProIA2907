# config.py
import os

# Rutas de archivos
MODEL_PATH = "lightgbm_model.pkl"
FEATURE_COLS_PATH = "feature_columns.pkl"

# Parámetros del modelo
LSTM_WINDOW_SIZE = 24   # ventana de retrasos previos (24 vuelos)
TOLERANCE_MINUTES = 15  # umbral de alerta IATA

# Puertas por terminal (según tus datos)
# NOTA: Las puertas reales pueden tener nombres como "G1", "A1", etc.
# Ajusta estas listas según el catálogo real de puertas
TERMINAL1_GATES = [str(i) for i in range(1, 37)]      # T1: puertas 1 a 36
TERMINAL2_GATES = [str(i) for i in range(52, 82)]    # T2: puertas 52 a 81

# Aerolíneas que operan en Terminal 2 (restricción)
TERMINAL2_AIRLINES = ["AMX", "DAL"]   # Aeroméxico y Delta (según tus datos)

# Conexión a SQL Server (autenticación Windows)
DB_SERVER = "localhost"
DB_NAME = "Aeropuerto"
DB_DRIVER = "ODBC Driver 17 for SQL Server"
DATABASE_URL = f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}?driver={DB_DRIVER}&trusted_connection=yes"

# Tabla principal
TABLE_NAME = "DatosAeropuerto2026EneroMayo"

# Intervalo de simulación (segundos)
POLLING_INTERVAL_SECONDS = 10