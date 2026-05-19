# Librerias 
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from streamlit_folium import st_folium
from gate_mapper import create_gate_map
from gate_optimizer import find_alternative_gate
from preprocess import compute_delays, add_time_features, create_lagged_features
from db_connector import load_all_flights
from config import MODEL_PATH, FEATURE_COLS_PATH, LSTM_WINDOW_SIZE, TOLERANCE_MINUTES, TERMINAL1_GATES, TERMINAL2_GATES

st.set_page_config(page_title="Sistema Inteligente de Control Operativo", layout="wide")
st.title(" Centro de Control Operativo ")
st.markdown("**Asignación inteligente de puertas** – Prediccion y Optimización de las operaciones")

# Cargar modelo y columnas
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(FEATURE_COLS_PATH)
    return model, cols
model, feature_cols = load_model()

# Cargar datos históricos completos (para construir ventanas de predicción)
@st.cache_data
def load_historical():
    df = load_all_flights()
    df = compute_delays(df)
    df = add_time_features(df)
    return df.sort_values('slot')
df_hist = load_historical()

# Inicializar estado de sesión
if 'active_flights' not in st.session_state:
    st.session_state.active_flights = []  # cada vuelo: {'id', 'vuelo', 'aerolinea', 'slot', 'delay', 'puerta', 'fin'}
if 'next_id' not in st.session_state:
    st.session_state.next_id = 1
if 'log' not in st.session_state:
    st.session_state.log = []
if 'simulation_time' not in st.session_state:
    # Inicializar tiempo virtual con el momento actual (se puede cambiar)
    st.session_state.simulation_time = datetime.now()

def predict_delay_for_flight(slot_time, aerolinea, terminal, tipo_avion):
    """Predice retraso usando el modelo y datos históricos anteriores a slot_time"""
    hist_before = df_hist[df_hist['slot'] < slot_time].copy()
    if len(hist_before) < LSTM_WINDOW_SIZE:
        return 0.0
    df_lag = create_lagged_features(hist_before, lags=LSTM_WINDOW_SIZE)
    if df_lag.empty:
        return 0.0
    last = df_lag.iloc[-1]
    X = last[feature_cols].values.reshape(1, -1)
    delay = model.predict(X)[0]
    return max(0, delay)

def liberar_puertas():
    """Elimina vuelos activos cuya hora_fin sea menor o igual al tiempo virtual actual"""
    nuevos_activos = []
    for v in st.session_state.active_flights:
        if v['fin'] > st.session_state.simulation_time:
            nuevos_activos.append(v)
        else:
            st.session_state.log.append(f"🟢 Puerta {v['puerta']} liberada (vuelo {v['vuelo']})")
    st.session_state.active_flights = nuevos_activos

def obtener_ocupacion_actual():
    ocupacion_dict = {gate: 'green' for gate in TERMINAL1_GATES + TERMINAL2_GATES}
    for v in st.session_state.active_flights:
        puerta = v.get('puerta')
        if puerta is not None:
            # Si es una Serie, extraer el primer valor
            if isinstance(puerta, pd.Series):
                puerta = puerta.iloc[0] if len(puerta) > 0 else ''
            puerta = str(puerta)
            if puerta in ocupacion_dict:
                ocupacion_dict[puerta] = 'red'
    if st.session_state.active_flights:
        ocup_df = pd.DataFrame([{
            'puerta': str(v.get('puerta', '')),
            'slot_inicio': v.get('slot'),
            'slot_fin': v.get('fin')
        } for v in st.session_state.active_flights])
    else:
        ocup_df = None
    return ocupacion_dict, ocup_df

def verificar_disponibilidad(puerta, slot_inicio, slot_fin, ocup_df):
    """Retorna True si la puerta está libre en el intervalo [slot_inicio, slot_fin]"""
    if ocup_df is None or ocup_df.empty:
        return True
    conflictos = ocup_df[(ocup_df['puerta'] == puerta) &
                         (ocup_df['slot_inicio'] < slot_fin) &
                         (ocup_df['slot_fin'] > slot_inicio)]
    return conflictos.empty

# --- Mostrar tiempo virtual actual ---
st.sidebar.header(" Tiempo virtual de simulación")
st.sidebar.markdown(f"**{st.session_state.simulation_time.strftime('%Y-%m-%d %H:%M')}**")
if st.sidebar.button(" Avanzar tiempo 1 hora"):
    st.session_state.simulation_time += timedelta(hours=1)
    liberar_puertas()
    st.rerun()

# --- Interfaz para añadir vuelo ---
st.sidebar.markdown("---")
st.sidebar.header(" Nuevo vuelo entrante")
with st.sidebar.form("nuevo_vuelo"):
    col1, col2 = st.columns(2)
    with col1:
        aerolinea = st.text_input("Aerolínea (ICAO)", "AMX")
        vuelo_num = st.text_input("Número de vuelo", "AMX123")
        terminal = st.selectbox("Terminal", ["1", "2"])
    with col2:
        tipo_avion = st.text_input("Tipo de avión", "B737")
        slot_time = st.datetime_input("Hora de slot", st.session_state.simulation_time + timedelta(minutes=30))
        puerta_asignada_inicial = st.selectbox("Puerta original (opcional)", ["(Ninguna)"] + TERMINAL1_GATES + TERMINAL2_GATES)
    
    submitted = st.form_submit_button(" Registrar vuelo entrante")

if submitted:
    # Liberar puertas según el tiempo virtual actual
    liberar_puertas()
    
    # Predecir retraso
    delay = predict_delay_for_flight(slot_time, aerolinea, terminal, tipo_avion)
    slot_fin_estimado = slot_time + timedelta(minutes=delay + 60)  # 60 min de estacionamiento
    
    # Determinar puerta original
    if puerta_asignada_inicial == "(Ninguna)":
        puerta_orig = TERMINAL1_GATES[0] if terminal == "1" else TERMINAL2_GATES[0]
    else:
        puerta_orig = puerta_asignada_inicial
    
    # Obtener ocupación actual
    _, ocup_df = obtener_ocupacion_actual()
    
    # Verificar si la puerta original está disponible en la ventana
    disponible = verificar_disponibilidad(puerta_orig, slot_time, slot_fin_estimado, ocup_df)
    
    puerta_final = None
    reasignado = False
    
    if disponible:
        puerta_final = puerta_orig
        st.session_state.log.append(f" Vuelo {vuelo_num} asignado a puerta {puerta_final} (retraso {delay:.0f} min)")
    else:
        # Buscar alternativa
        vuelo_dict = {
            'slot': slot_time,
            'aerolinea': aerolinea,
            'puerta_original': puerta_orig,
            'terminal': terminal
        }
        nueva_puerta = find_alternative_gate(vuelo_dict, delay, ocup_df)
        if nueva_puerta:
            puerta_final = nueva_puerta
            reasignado = True
            st.session_state.log.append(f" Puerta {puerta_orig} ocupada. Reasignado a {puerta_final} (retraso {delay:.0f} min)")
        else:
            puerta_final = puerta_orig
            st.session_state.log.append(f" CONFLICTO: Puerta {puerta_orig} ocupada y no hay alternativa. Asignado igual.")
    
    # Registrar vuelo activo
    nuevo_vuelo = {
        'id': st.session_state.next_id,
        'vuelo': vuelo_num,
        'aerolinea': aerolinea,
        'slot': slot_time,
        'delay': round(delay, 1),
        'puerta': puerta_final,
        'fin': slot_fin_estimado,
        'original': puerta_orig,
        'reasignado': reasignado
    }
    st.session_state.active_flights.append(nuevo_vuelo)
    st.session_state.next_id += 1
    
    # Mostrar mensaje de éxito en la interfaz
    if reasignado:
        st.success(f" Vuelo {vuelo_num} reasignado a puerta **{puerta_final}** (original era {puerta_orig})")
    else:
        st.success(f" Vuelo {vuelo_num} asignado a puerta **{puerta_final}**")
    st.rerun()

# --- Mostrar estado actual ---
tab1, tab2 = st.tabs([" Mapa de puertas", " Vuelos activos"])

with tab1:
    estado_map, _ = obtener_ocupacion_actual()
    # Marcar la última puerta asignada en amarillo
    if st.session_state.active_flights:
        ultimo = st.session_state.active_flights[-1]
        if ultimo['puerta'] in estado_map:
            estado_map[ultimo['puerta']] = 'yellow'
    m = create_gate_map(estado_map)
    st_folium(m, width=900, height=500)

with tab2:
    if st.session_state.active_flights:
        df_act = pd.DataFrame(st.session_state.active_flights)
        df_act['slot'] = df_act['slot'].dt.strftime("%H:%M %d/%m")
        df_act['fin'] = df_act['fin'].dt.strftime("%H:%M")
        df_act = df_act[['vuelo', 'aerolinea', 'slot', 'delay', 'puerta', 'fin', 'original', 'reasignado']]
        df_act.rename(columns={
            'vuelo': 'Vuelo', 'aerolinea': 'Aerolínea', 'slot': 'Slot',
            'delay': 'Retraso (min)', 'puerta': 'Puerta asignada', 'fin': 'Libre hasta',
            'original': 'Puerta original', 'reasignado': 'Reasignado?'
        }, inplace=True)
        st.dataframe(df_act, use_container_width=True)
    else:
        st.info("No hay vuelos activos. Agrega un vuelo desde el panel lateral.")

# --- Log de eventos ---
st.sidebar.markdown("---")
st.sidebar.subheader(" Registro")
if st.sidebar.button(" Limpiar log"):
    st.session_state.log = []
for msg in st.session_state.log[-10:]:
    st.sidebar.text(msg)
# ========== SIMULACIÓN AUTOMÁTICA CON DATOS HISTÓRICOS (30%) ==========
st.sidebar.markdown("---")
st.sidebar.subheader(" Simulación automática (30% datos)")
if st.sidebar.button(" Iniciar simulación histórica (sin mapa)"):
    import time
    import pandas as pd
    import traceback
    from datetime import timedelta

    try:
        df_sim = pd.read_csv('datos_simulacion.csv', parse_dates=['slot'])
        st.info(f"Columnas originales: {list(df_sim.columns)}")

        # 1. Eliminar columnas que causan conflicto (fechas de posición)
        columnas_a_eliminar = ['asa_FechaHoraPosicion2', 'posicion2']
        df_sim.drop(columns=[c for c in columnas_a_eliminar if c in df_sim.columns], errors='ignore', inplace=True)

        # 2. Renombrar columnas usando nombres exactos (sin subcadenas)
        rename_map = {
            'asa_NoVuelo': 'vuelo',
            'asa_AerolineaICAO': 'aerolinea',
            'asa_Posicion': 'puerta_original',
            'asa_Terminal': 'terminal',
            'asa_TipoAvion': 'tipo_avion'
        }
        df_sim.rename(columns=rename_map, inplace=True)

        # 3. Verificar que existan las columnas necesarias
        required = ['vuelo', 'aerolinea', 'puerta_original', 'terminal', 'slot', 'delay_to_gate']
        missing = [c for c in required if c not in df_sim.columns]
        if missing:
            st.error(f"Faltan columnas: {missing}. Columnas disponibles: {list(df_sim.columns)}")
            st.stop()

        # 4. Convertir puerta a string y reconstruir posicion2 (coherente)
        df_sim['puerta_original'] = df_sim['puerta_original'].astype(str)
        df_sim['posicion2'] = df_sim['slot'] + pd.to_timedelta(df_sim['delay_to_gate'], unit='m')

        # 5. Ordenar y tomar un subconjunto (ajusta el número)
        df_sim = df_sim.sort_values('slot').reset_index(drop=True)
        df_sim = df_sim.head(30)  # puedes aumentar para pruebas más largas

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.code(traceback.format_exc())
        st.stop()

    # Reiniciar estado de la sesión
    st.session_state.active_flights = []
    st.session_state.log = []
    st.session_state.next_id = 1
    st.session_state.simulation_time = datetime.now()

    progress_bar = st.progress(0)
    status_text = st.empty()
    table_placeholder = st.empty()
    log_placeholder = st.empty()

    total = len(df_sim)
    for idx, row in df_sim.iterrows():
        try:
            st.session_state.simulation_time = row['slot']
            liberar_puertas()

            # Datos del vuelo
            slot = row['slot']
            aerolinea = row['aerolinea']
            vuelo_num = row['vuelo']
            terminal = row['terminal']
            tipo_avion = row.get('tipo_avion', 'B737')
            puerta_orig = row['puerta_original']  # ahora es un string, ej. "25"

            # Predicción de retraso
            delay = predict_delay_for_flight(slot, aerolinea, terminal, tipo_avion)
            slot_fin_estimado = slot + timedelta(minutes=delay + 60)

            # Obtener ocupación actual
            _, ocup_df = obtener_ocupacion_actual()
            disponible = verificar_disponibilidad(puerta_orig, slot, slot_fin_estimado, ocup_df)

            if disponible:
                puerta_final = puerta_orig
                reasignado = False
                msg = f"✅ {vuelo_num} -> puerta {puerta_final} (retraso {delay:.0f} min)"
            else:
                vuelo_dict = {
                    'slot': slot,
                    'aerolinea': aerolinea,
                    'puerta_original': puerta_orig,
                    'terminal': terminal
                }
                nueva_puerta = find_alternative_gate(vuelo_dict, delay, ocup_df)
                if nueva_puerta:
                    puerta_final = str(nueva_puerta)
                    reasignado = True
                    msg = f"⚠️ {vuelo_num}: puerta {puerta_orig} ocupada -> reasignado a {puerta_final} (retraso {delay:.0f} min)"
                else:
                    puerta_final = puerta_orig
                    msg = f"❌ {vuelo_num}: CONFLICTO sin alternativa en puerta {puerta_orig}"

            # Registrar vuelo activo
            nuevo_vuelo = {
                'id': st.session_state.next_id,
                'vuelo': vuelo_num,
                'aerolinea': aerolinea,
                'slot': slot,
                'delay': round(delay, 1),
                'puerta': puerta_final,
                'fin': slot_fin_estimado,
                'original': puerta_orig,
                'reasignado': reasignado
            }
            st.session_state.active_flights.append(nuevo_vuelo)
            st.session_state.next_id += 1
            st.session_state.log.append(msg)

            # Mostrar tabla actualizada
            df_act = pd.DataFrame(st.session_state.active_flights)
            df_act['slot'] = df_act['slot'].dt.strftime("%H:%M")
            df_act['fin'] = df_act['fin'].dt.strftime("%H:%M")
            df_act = df_act[['vuelo', 'aerolinea', 'slot', 'delay', 'puerta', 'fin', 'original', 'reasignado']]
            df_act.rename(columns={
                'vuelo': 'Vuelo', 'aerolinea': 'Aerolínea', 'slot': 'Slot',
                'delay': 'Retraso (min)', 'puerta': 'Puerta asignada', 'fin': 'Libre hasta',
                'original': 'Puerta original', 'reasignado': 'Reasignado?'
            }, inplace=True)
            table_placeholder.dataframe(df_act, use_container_width=True)

            # Mostrar log
            log_placeholder.text_area("Eventos recientes", "\n".join(st.session_state.log[-10:]), height=150)

            status_text.text(f"Procesando {idx+1}/{total}: {vuelo_num}")
            progress_bar.progress((idx+1)/total)
            time.sleep(0.3)  # velocidad entre vuelos

        except Exception as e:
            st.error(f"Error en vuelo {idx}: {e}")
            st.code(traceback.format_exc())
            break
    else:
        st.success("Simulación completada")