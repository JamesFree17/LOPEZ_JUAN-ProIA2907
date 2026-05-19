#LOPEZ_JUAN-ProIA2907
# db_connector.py
import pandas as pd
from sqlalchemy import create_engine, text
from config import DATABASE_URL, TABLE_NAME

engine = create_engine(DATABASE_URL)

def load_all_flights():
    """Carga todos los registros de la tabla principal (ordenados por slot)."""
    query = f"""
        SELECT 
            asa_NoVuelo,
            asa_AerolineaICAO,
            asa_FechaHoraSlot,
            asa_FechaHoraPosicion2,
            asa_TipoAvion,
            asa_Terminal,
            asa_Posicion,
            asa_TipoOperacion
        FROM {TABLE_NAME}
        WHERE asa_FechaHoraSlot IS NOT NULL 
          AND asa_FechaHoraPosicion2 IS NOT NULL
        ORDER BY asa_FechaHoraSlot
    """
    df = pd.read_sql(query, engine)
    return df

def get_historical_data_before(cutoff_date):
    """Obtiene datos históricos con slot anterior a una fecha dada."""
    query = f"""
        SELECT 
            asa_NoVuelo,
            asa_AerolineaICAO,
            asa_FechaHoraSlot,
            asa_FechaHoraPosicion2,
            asa_TipoAvion,
            asa_Terminal,
            asa_Posicion,
            asa_TipoOperacion
        FROM {TABLE_NAME}
        WHERE asa_FechaHoraSlot < ?
          AND asa_FechaHoraSlot IS NOT NULL 
          AND asa_FechaHoraPosicion2 IS NOT NULL
        ORDER BY asa_FechaHoraSlot
    """
    df = pd.read_sql(query, engine, params=(cutoff_date,))
    return df