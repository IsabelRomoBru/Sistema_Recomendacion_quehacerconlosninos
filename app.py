import streamlit as st
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"

from sentence_transformers import SentenceTransformer
import torch

modelo_nlp = SentenceTransformer('./model_cache')


# Configuración general de la app
st.set_page_config(
    page_title="¿Qué hacer con los niños?",
    page_icon="favicon.png",
    layout="centered"
)
# Estilo 
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
        background-color: #BDDCF2;
        color: #0277AE;
    }

    .stApp {
        background-color: #BDDCF2;
    }

    h1, h2, h3 {
        color: #0277AE;
    }

    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }

    .stSidebar {
        background-color: #FFC0D2;
    }

    /* ✅ Eliminar recuadro blanco en expanders y cajas */
    .st-expander, .stAlert, .st-cx, .stContainer {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ✅ Ocultar marco de widgets si existiera */
    div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)


# Mostrar logo principal
st.image("logo-princi.png", width=350)

# Título principal
st.title("🎈 ¿Qué hacer con los niños?")
st.subheader("Encuentra actividades inclusivas y divertidas para tu familia")

# Cargar datos
data = pd.read_csv('actividades_familiares_españa_v3.csv')

# Limpiar columnas
data['necesidad_especial'] = data['necesidad_especial'].fillna('No especificado')
data['precio'] = data['precio'].fillna('No especificado')
data['categoria'] = data['categoria'].fillna('Sin categoría')
data['descripcion'] = data['descripcion'].fillna('')

# Panel lateral (filtros)
st.sidebar.header("💬 Tus preferencias")
edad_usuario = st.sidebar.slider('Edad del niño/a', 1, 16, 6)
ciudad_usuario = st.sidebar.selectbox('Ubicación', options=data['ubicacion'].dropna().unique())

opciones_diversidad = data['necesidad_especial'].dropna().unique()
diversidad_funcional = st.sidebar.selectbox('Diversidad funcional', options=['Todos'] + list(opciones_diversidad))

opciones_precio = data['precio'].dropna().unique()
filtro_precio = st.sidebar.selectbox('Tipo de precio', options=['Todos'] + list(opciones_precio))

opciones_categoria = data['categoria'].dropna().unique()
filtro_categoria = st.sidebar.selectbox('Categoría de actividad', options=['Todas'] + list(opciones_categoria))

consulta_texto = st.sidebar.text_input('Describe qué te gustaría hacer:', 'Actividad tranquila en casa')

# Estado de sesión para actividades guardadas
if 'actividades_guardadas' not in st.session_state:
    st.session_state.actividades_guardadas = []

# Botón de búsqueda
if st.sidebar.button('🔍 Buscar actividades'):
    # Aplicar filtros
    filtro = (data['edad_minima'] <= edad_usuario) & (data['ubicacion'] == ciudad_usuario)

    if diversidad_funcional != 'Todos':
        filtro &= data['necesidad_especial'].str.lower().str.contains(diversidad_funcional.lower(), na=False)

    if filtro_precio != 'Todos':
        filtro &= data['precio'].str.lower().str.contains(filtro_precio.lower(), na=False)

    if filtro_categoria != 'Todas':
        filtro &= (data['categoria'] == filtro_categoria)

    actividades_filtradas = data[filtro].reset_index(drop=True)

    if actividades_filtradas.empty:
        st.error("⚠️ No se encontraron actividades que cumplan con tus criterios. Prueba cambiando algún filtro.")
    else:
        # Calcular similitud semántica
        descripciones = actividades_filtradas['descripcion'].tolist()
        embeddings_actividades = modelo_nlp.encode(descripciones, convert_to_tensor=True)
        embedding_consulta = modelo_nlp.encode([consulta_texto], convert_to_tensor=True)

        similitudes = cosine_similarity(embedding_consulta, embeddings_actividades)[0]
        actividades_filtradas['similitud'] = similitudes

        resultados = actividades_filtradas.sort_values(by='similitud', ascending=False)
        resultados_unicos = resultados.drop_duplicates(subset='nombre_evento').reset_index(drop=True)

        top_3 = resultados_unicos.head(3)

        # 🎯 ACTIVIDAD RECOMENDADA - TARJETA ROJA
        actividad_recomendada = top_3.iloc[0]
        st.markdown(f"""
            <div style="background-color:#FFC0D2; border-radius:15px; padding:1.5em; border: 3px solid #FF6B6B; margin-bottom: 2em;">
                <h2 style="color:#0277AE;">🎯 Actividad recomendada: {actividad_recomendada['nombre_evento']}</h2>
                <p style="color:#444;"><em>{actividad_recomendada['descripcion']}</em></p>
                <p><strong>Categoría:</strong> {actividad_recomendada['categoria']}<br>
                   <strong>Precio:</strong> {actividad_recomendada['precio']}<br>
                   <strong>Accesibilidad:</strong> {actividad_recomendada['necesidad_especial']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Botones recomendada
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("🔍 Ver más detalles"):
                st.markdown(f"- 🗓️ **Fecha**: {actividad_recomendada.get('fecha', 'No disponible')}")
                st.markdown(f"- 📍 **Ubicación**: {actividad_recomendada.get('ubicacion', 'No especificada')}")
                st.markdown(f"- 📝 **Descripción completa**:\n\n{actividad_recomendada['descripcion']}")
        with col2:
            if st.button(f"💾 Guardar: {actividad_recomendada['nombre_evento']}"):
                st.session_state.actividades_guardadas.append(actividad_recomendada['nombre_evento'])
                st.success("Actividad guardada.")

        # 📘 SUGERENCIAS
        if len(top_3) > 1:
            st.markdown("### 📘 Otras opciones que podrían interesarte:")
            for i in range(1, min(3, len(top_3))):
                actividad = top_3.iloc[i]
                st.markdown(f"""
                    <div style="background-color:#BDDCF2; border-radius:12px; padding:1.2em; margin-bottom:1.5em; border: 2px solid #0277AE;">
                        <h3 style="color:#FF6B6B;">🎯 {actividad['nombre_evento']}</h3>
                        <p style="color:#444;"><em>{actividad['descripcion']}</em></p>
                        <p><strong>Categoría:</strong> {actividad['categoria']}<br>
                           <strong>Precio:</strong> {actividad['precio']}<br>
                           <strong>Accesibilidad:</strong> {actividad['necesidad_especial']}</p>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    with st.expander(f"🔍 Ver más sobre: {actividad['nombre_evento']}"):
                        st.markdown(f"- 🗓️ **Fecha**: {actividad.get('fecha', 'No disponible')}")
                        st.markdown(f"- 📍 **Ubicación**: {actividad.get('ubicacion', 'No especificada')}")
                        st.markdown(f"- 📝 **Descripción completa**:\n\n{actividad['descripcion']}")
                with col2:
                    if st.button(f"💾 Guardar: {actividad['nombre_evento']}", key=f"guardar_{i}"):
                        st.session_state.actividades_guardadas.append(actividad['nombre_evento'])
                        st.success("Actividad guardada.")
        else:
            st.info("Solo encontramos una actividad que se ajuste bien a tu búsqueda.")

        # Mostrar actividades guardadas
        if st.session_state.actividades_guardadas:
            st.markdown("### 📌 Actividades guardadas")
            for actividad in st.session_state.actividades_guardadas:
                st.write(f"- {actividad}")
