import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium

# ----------------------------
# Cargar datos
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("df_topsis_normalizado.csv")  # o .parquet si prefieres
    geo = gpd.read_file("Alicante_Barrios.json")  # si usas mapas
    geo = geo.rename(columns={'name': 'Barrio'})
    return df, geo

df, gdf = load_data()

# ----------------------------
# Función TOPSIS
# ----------------------------
def run_topsis(M: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    V = M * pesos
    ideal_pos = V.max(axis=0)
    ideal_neg = V.min(axis=0)
    d_pos = np.linalg.norm(V - ideal_pos, axis=1)
    d_neg = np.linalg.norm(V - ideal_neg, axis=1)
    return d_neg / (d_pos + d_neg)

# ----------------------------
# Interfaz de usuario
# ----------------------------
st.title("Descubre tu Barrio Ideal:")
st.markdown("###  Explora, Personaliza e Invierte en Alicante")


# Elegir DESTINO
destino = st.selectbox("¡Dale forma a tu FUTURO!: Escoge tu oportunidad de inversión", df['DESTINO'].unique())

# Sliders para los pesos
cols_directo = ['cantidad_ocio_comercio', 'cantidad_oficinas', 'cantidad_residencial', 'centros_docentes', 'nivel_socio_economico']
cols_inverso = ['distancia_al_mar', 'Valor/m2_predicho']
all_cols = cols_directo + cols_inverso

nombres_variables = {
    'cantidad_ocio_comercio': '🍽️🛍️ Porcentaje de Ocio y Comercio',
    'cantidad_oficinas': '🏢 Porcentaje de Oficinas',
    'cantidad_residencial': '🏘️ Porcentaje de Viviendas',
    'centros_docentes': '🏫 Cantidad de Centros Educativos',
    'nivel_socio_economico': '💰 Nivel Socioeconómico',
    'distancia_al_mar': '🌊 Cercanía al Mar',
    'Valor/m2_predicho': '💸 Precio por m² (estimado)'
}


st.subheader("⚖️ Personaliza tus prioridades: define la importancia de cada factor")

pesos = []
for col in cols_directo:
    label = nombres_variables.get(col, col)
    pesos.append(st.slider(f"{label}", -5, 5, 1))

for col in cols_inverso:
    label = nombres_variables.get(col, col)
    pesos.append(st.slider(f"{label}", -5, 5, -1))

pesos = np.array(pesos)

# ----------------------------
# Filtrar y calcular TOPSIS
# ----------------------------
df_filtrado = df[df["DESTINO"] == destino].copy()

# Normalizar
scaler = MinMaxScaler()
df_filtrado[cols_directo] = scaler.fit_transform(df_filtrado[cols_directo])
df_filtrado[cols_inverso] = scaler.fit_transform(df_filtrado[cols_inverso])
df_filtrado[cols_inverso] = 1 - df_filtrado[cols_inverso]

# Calcular TOPSIS
M = df_filtrado[all_cols].values
df_filtrado["topsis_score"] = run_topsis(M, pesos)

# ----------------------------
# Mostrar top 3
# ----------------------------
top3 = df_filtrado.sort_values("topsis_score", ascending=False).head(3)
st.subheader("🏆 Top 3 barrios recomendados")
# Renombrar columna, resetear índice y empezar desde 1
df_top3 = top3[['Barrio', 'topsis_score']].rename(columns={'topsis_score': 'Puntaje de Afinidad'}).reset_index(drop=True)
df_top3.index = df_top3.index + 1
st.dataframe(df_top3)


# ----------------------------
# Mapa con colores
# ----------------------------
st.subheader("🗺️ Alicante a color: Mapa de oportunidades")

# Unir con geometría
gdf_destino = gdf.merge(df_filtrado[['Barrio', 'topsis_score']], on='Barrio', how='left')

# Asignar colores
colores = ['red', 'orange', 'green']
color_map = {}

for i, (_, row) in enumerate(top3.iterrows()):
    color_map[row['Barrio']] = colores[i]

def get_color(barrio):
    return color_map.get(barrio, '#CCCCCC')

gdf_destino['color'] = gdf_destino['Barrio'].apply(get_color)

# Crear mapa
m = folium.Map(location=[38.3452, -0.4810], zoom_start=13)

for _, row in gdf_destino.iterrows():
    folium.GeoJson(
        row['geometry'],
        style_function=lambda feature, color=row['color']: {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=row['Barrio']
    ).add_to(m)

# Mostrar en Streamlit
st_folium(m, width=700, height=500)
