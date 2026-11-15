"""
app.py
Tablero interactivo + predictor de gravedad de siniestros viales (CABA 2019-2025).

La lógica de limpieza y modelado vive en siniestros_core.py.
"""
import io

import pandas as pd
import streamlit as st

import siniestros_core as core

st.set_page_config(page_title="Siniestros viales CABA", layout="wide")


# CARGA (con caché para que no se repita en cada interacción)
@st.cache_data(show_spinner="Cargando y limpiando datos...")
def cargar(hechos, victimas):
    return core.limpiar_hechos(hechos), core.limpiar_victimas(victimas)


@st.cache_resource(show_spinner="Entrenando modelo...")
def modelo_entrenado(_h, clave):
    # '_h' no se hashea (es grande); 'clave' invalida la caché si cambian los filtros
    return core.entrenar_modelo(_h)


st.sidebar.header("Datos")
f_hechos = st.sidebar.file_uploader("Hechos (.xlsx)", type="xlsx")
f_vict = st.sidebar.file_uploader("Víctimas (.csv)", type="csv")
ruta_h = io.BytesIO(f_hechos.read()) if f_hechos else "siniestros_viales_hechos.xlsx"
ruta_v = io.BytesIO(f_vict.read()) if f_vict else "siniestros_viales_victimas.csv"

try:
    hechos, victimas = cargar(ruta_h, ruta_v)
except FileNotFoundError:
    st.error("No se encontraron los archivos. Ponelos junto a este script "
             "o subilos desde la barra lateral.")
    st.stop()

# FILTROS GLOBALES
st.sidebar.header("Filtros")
anios = sorted(hechos["anio"].dropna().astype(int).unique())
rango = st.sidebar.slider("Años", min(anios), max(anios), (min(anios), max(anios)))
modos = sorted(victimas["modo"].dropna().unique())
modos_sel = st.sidebar.multiselect("Modo de desplazamiento de la víctima", modos,
                                   default=modos)

v = victimas[victimas["anio"].between(*rango)]
v = v[v["modo"].isin(modos_sel) | v["modo"].isna()]
h = hechos[hechos["anio"].between(*rango)]

st.title("Siniestralidad vial en la Ciudad de Buenos Aires")
st.caption("Fuente: datos abiertos de siniestros viales (hechos y víctimas), 2019-2025.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Panorama general", "Hipótesis: motos", "Predictor de riesgo", "Mapa y zonas"])

# TAB 1 - PANORAMA
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Siniestros", f"{len(h):,}".replace(",", "."))
    c2.metric("Víctimas", f"{len(v):,}".replace(",", "."))
    c3.metric("Víctimas graves", f"{(v.gravedad_victima == 'GRAVE').sum():,}".replace(",", "."))
    c4.metric("Víctimas fatales", f"{(v.gravedad_victima == 'MORTAL').sum():,}".replace(",", "."))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Víctimas por año y gravedad")
        t = v.groupby(["anio", "gravedad_victima"]).size().unstack(fill_value=0)
        st.bar_chart(t)
    with col_b:
        st.subheader("% de siniestros graves/mortales por hora")
        por_hora = h.groupby("hora")["grave_o_mortal"].mean() * 100
        st.line_chart(por_hora)
        st.caption("La madrugada (0-5 h) concentra la mayor proporción de casos graves.")

    st.subheader("Gravedad según modo de desplazamiento")
    tm = core.tabla_por_modo(v).round(2)
    st.dataframe(tm.rename(columns={
        "victimas": "Víctimas", "mortales": "Fatales", "graves": "Graves",
        "pct_grave_mortal": "% grave o fatal", "pct_mortal": "% fatal"}),
        width="stretch")

# TAB 2 - HIPÓTESIS
with tab2:
    st.subheader("H1: los motociclistas sufren más lesiones graves/fatales que los ocupantes de auto")
    st.markdown("Test chi-cuadrado de independencia sobre víctimas con modo MOTO o AUTO "
                "(respeta el filtro de años de la barra lateral).")
    try:
        r = core.test_hipotesis_moto(victimas[victimas["anio"].between(*rango)])
        a, b, c = st.columns(3)
        a.metric("% grave/fatal - MOTO", f"{r['p_moto'] * 100:.1f}%")
        b.metric("% grave/fatal - AUTO", f"{r['p_auto'] * 100:.1f}%")
        c.metric("Razón de proporciones", f"{r['razon']:.2f}x")
        st.write(f"χ² = {r['chi2']:.1f} · p-valor = {r['p_valor']:.2e}")
        if r["p_valor"] < 0.05:
            st.success("Se rechaza H0: la diferencia es estadísticamente significativa.")
        else:
            st.warning("No hay evidencia suficiente para rechazar H0 en este rango.")
        st.dataframe(r["tabla"].rename(columns={False: "Leve", True: "Grave/Fatal"}))
    except Exception as e:  # p. ej. si el rango deja sin datos
        st.info(f"No se pudo calcular el test con la selección actual ({e}).")

# TAB 3 - PREDICTOR
with tab3:
    st.subheader("Estimá el riesgo de un siniestro grave o fatal")
    st.markdown("Random Forest entrenado con los hechos históricos. Devuelve la "
                "**probabilidad** de que el siniestro tenga al menos una víctima "
                "grave o fatal, dado el contexto.")
    modelo, met = modelo_entrenado(hechos, "completo")

    with st.expander("Desempeño del modelo (datos de prueba, 25 %)"):
        m1, m2, m3 = st.columns(3)
        m1.metric("AUC-ROC", f"{met['auc']:.3f}")
        m2.metric("Recall (graves/fatales)", f"{met['recall_grave']:.0%}")
        m3.metric("Tasa base de casos graves", f"{met['base_rate']:.1%}")
        st.write("Importancia de variables")
        st.bar_chart(met["importancias"])
        st.caption("El modelo prioriza no dejar pasar casos graves (recall) a costa de "
                   "más falsas alarmas: sirve para **priorizar**, no para decidir solo.")

    c1, c2, c3 = st.columns(3)
    modo = c1.selectbox("Modo de la víctima", sorted(hechos["modo_desplazamiento_victima"].dropna().unique()),
                        index=0)
    contra = c2.selectbox("Contraparte", sorted(hechos["contraparte_siniestro"].dropna().unique()))
    via = c3.selectbox("Tipo de vía", sorted(hechos["tipo_via"].dropna().unique()))
    c4, c5, c6, c7 = st.columns(4)
    hora = c4.slider("Hora", 0, 23, 22)
    dia = c5.selectbox("Día", list(core.DIAS.keys()), format_func=core.DIAS.get, index=5)
    comuna = c6.selectbox("Comuna", list(range(1, 16)))
    mes = c7.slider("Mes", 1, 12, 6)

    if st.button("Calcular riesgo", type="primary"):
        base = met["base_rate"]
        p = core.predecir(modelo, modo, contra, via, hora, dia, comuna, mes, prior=base)
        st.metric("Probabilidad de siniestro grave/fatal", f"{p:.1%}",
                  delta=f"{(p / base - 1):+.0%} vs. promedio ({base:.1%})",
                  delta_color="inverse")

# TAB 4 - MAPA Y ZONAS
with tab4:
    st.subheader("Dónde ocurren los siniestros graves y fatales")

    # Controles del mapa
    c_a, c_b = st.columns([2, 1])
    estilo = c_a.radio("Tipo de mapa", ["Animado por hora", "Puntos por gravedad", "Mapa de calor"],
                       horizontal=True)
    solo_fatales = c_b.checkbox("Mostrar solo fatales")

    # Datos del mapa: solo hechos con coordenadas válidas
    cols = ["latitud_siniestro", "longitud_siniestro", "gravedad",
            "direccion_normalizada_siniestro", "fecha", "modo_desplazamiento_victima", "hora"]
    pts = h[h["grave_o_mortal"] == 1][cols].dropna(subset=cols[:2]).copy()
    if solo_fatales:
        pts = pts[pts["gravedad"] == "MORTAL"]
    pts = pts.rename(columns={"latitud_siniestro": "lat", "longitud_siniestro": "lon",
                              "direccion_normalizada_siniestro": "direccion",
                              "modo_desplazamiento_victima": "modo"})
    pts_anim = pts.copy()   # copia con hora y fecha para el mapa animado
    # El tooltip no admite valores nulos: se reemplazan por texto
    pts["direccion"] = pts["direccion"].fillna("Sin dato")
    pts["modo"] = pts["modo"].fillna("Sin dato")
    pts["fecha_txt"] = pts["fecha"].dt.strftime("%d/%m/%Y").fillna("Sin dato")
    # Color por gravedad: fatal = rojo oscuro, grave = naranja
    colores = {"MORTAL": [180, 30, 30, 200], "GRAVE": [240, 140, 40, 150]}
    pts["color"] = pts["gravedad"].map(colores)
    pts = pts[["lat", "lon", "gravedad", "direccion", "fecha_txt", "modo", "color"]]

    # Construcción del mapa
    if estilo == "Animado por hora":
        # Mapa animado (deck.gl) incrustado como HTML: ver mapa_animado.py
        from mapa_animado import html_mapa_animado
        html_mapa = html_mapa_animado(pts_anim)
        if hasattr(st, "iframe"):
            # Streamlit reciente: components.v1.html quedó en desuso
            st.iframe(html_mapa, height=680)
        else:
            import streamlit.components.v1 as components
            components.html(html_mapa, height=680)
        st.caption("Cada punto es un siniestro grave o fatal; el reloj recorre las 24 horas "
                   "con todos los años superpuestos. Hacé clic en una barra para saltar a esa hora. "
                   "Necesita conexión a internet para cargar el mapa base.")
    else:
        import pydeck as pdk   # viene incluido con Streamlit

        if estilo == "Mapa de calor":
            capa = pdk.Layer("HeatmapLayer", data=pts, get_position="[lon, lat]",
                             get_weight=1, radius_pixels=35, opacity=0.8)
            tooltip = None
        else:
            capa = pdk.Layer("ScatterplotLayer", data=pts, get_position="[lon, lat]",
                             get_fill_color="color", get_radius=60,
                             radius_min_pixels=3, radius_max_pixels=12, pickable=True)
            tooltip = {"html": "<b>{gravedad}</b><br/>{direccion}<br/>{fecha_txt} · {modo}"}

        vista = pdk.ViewState(latitude=-34.6118, longitude=-58.4173, zoom=10.6)
        st.pydeck_chart(pdk.Deck(layers=[capa], initial_view_state=vista,
                                 tooltip=tooltip, map_style="light"), height=520)

        if estilo == "Puntos por gravedad":
            st.markdown(
                "<span style='color:#B41E1E'>&#9679;</span> Fatal &nbsp;&nbsp; "
                "<span style='color:#F08C28'>&#9679;</span> Grave &nbsp;&nbsp; "
                "(pasá el cursor por un punto para ver dirección y fecha)",
                unsafe_allow_html=True)
        st.caption(f"{len(pts):,} puntos (se omiten los hechos sin coordenadas válidas).".replace(",", "."))

    # Comparación por comuna
    st.subheader("Siniestros y víctimas fatales por comuna")
    tc = core.tabla_por_comuna(h)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Siniestros por comuna**")
        st.bar_chart(tc["siniestros"], color="#3A6EA5")
    with g2:
        st.markdown("**Víctimas fatales por comuna**")
        st.bar_chart(tc["mortales"], color="#B41E1E")
    st.caption("Cerca del 5 % de los siniestros no tiene comuna informada y no aparece en estos gráficos.")