#LOPEZ_JUAN-ProIA2907
# gate_optimizer.py
from config import TERMINAL1_GATES, TERMINAL2_GATES, TERMINAL2_AIRLINES, TOLERANCE_MINUTES
import pandas as pd
from datetime import timedelta

def find_alternative_gate(vuelo, delay_predicho, ocupacion_df=None):
    """
    vuelo: dict con 'slot', 'aerolinea', 'puerta_original', 'terminal'
    ocupacion_df: DataFrame con columnas 'puerta', 'slot_inicio', 'slot_fin'
    Retorna: puerta sugerida o None
    """
    slot_original = vuelo['slot']
    nueva_hora_fin = slot_original + timedelta(minutes=delay_predicho + TOLERANCE_MINUTES)
    
    # Determinar puertas candidatas según terminal
    if vuelo['terminal'] == '2':
        candidatas = TERMINAL2_GATES.copy()
    else:
        candidatas = TERMINAL1_GATES.copy()
    
    # Excluir la puerta original
    if vuelo['puerta_original'] in candidatas:
        candidatas.remove(vuelo['puerta_original'])
    
    # Si no hay información de ocupación, devolver la primera disponible
    if ocupacion_df is None or ocupacion_df.empty:
        return candidatas[0] if candidatas else None
    
    # Convertir a datetime si no lo están
    ocupacion_df = ocupacion_df.copy()
    ocupacion_df['slot_inicio'] = pd.to_datetime(ocupacion_df['slot_inicio'])
    ocupacion_df['slot_fin'] = pd.to_datetime(ocupacion_df['slot_fin'])
    
    # Puertas ocupadas en la ventana
    ocupadas = set(ocupacion_df[
        (ocupacion_df['slot_inicio'] < nueva_hora_fin) & 
        (ocupacion_df['slot_fin'] > slot_original)
    ]['puerta'].unique())
    
    for gate in candidatas:
        if gate not in ocupadas:
            return gate
    return None