import streamlit as st
from supabase import create_client, Client
import pandas as pd

SUPABASE_URL = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]

@st.cache_resource
def init_connection() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def load_all_flights():
    supabase = init_connection()
    response = supabase.table('DatosAeropuerto2026EneroMayo')\
        .select('*')\
        .limit(1)\
        .execute()
    df = pd.DataFrame(response.data)
    print("🔍 COLUMNAS REALES en Supabase:", df.columns.tolist())
    # Forzar conversión de fechas solo si las columnas existen
    for col in ['asa_FechaHoraSlot', 'asa_FechaHoraPosicion2', 'asa_FechaHoraAterrizaje']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def get_historical_data_before(cutoff_date):
    supabase = init_connection()
    cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
    # Usar filtro de fecha
    response = supabase.table('DatosAeropuerto2026EneroMayo')\
        .select('*')\
        .lt('asa_FechaHoraSlot', cutoff_str)\
        .execute()
    df = pd.DataFrame(response.data)
    for col in ['asa_FechaHoraSlot', 'asa_FechaHoraPosicion2']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df