"""
siniestros_core.py
Funciones de carga, limpieza, análisis y modelado del dataset de
siniestros viales de la Ciudad Autónoma de Buenos Aires (2019-2025).

"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Nombres de días y orden para gráficos
DIAS = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
        4: "Viernes", 5: "Sábado", 6: "Domingo"}


# 1. CARGA Y LIMPIEZA
def limpiar_hechos(ruta_xlsx: str) -> pd.DataFrame:
    """Lee la hoja HECHOS y normaliza los campos problemáticos."""
    h = pd.read_excel(ruta_xlsx)


    h = h.dropna(subset=["id_siniestro"]).drop_duplicates(subset=["id_siniestro"]).copy()

    # Conteos de víctimas: en 2025 hay ~3.300 siniestros con 'SD' (aún sin
    # procesar). Se dejan como NaN (no como 0) para no subestimar los totales;
    # para totales anuales conviene usar la tabla de víctimas.
    for col in ["numero_total_de_victimas", "numero_victimas_leve_siniestro",
                "numero_victimas_grave_siniestro", "numero_victimas_mortal_siniestro"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")

    # 'comuna_siniestro' viene mezclada: "Comuna 1", "COMUNA 1", "1", "SD"
    c = (h["comuna_siniestro"].astype(str).str.upper()
         .str.replace("COMUNA", "", regex=False).str.strip())
    h["comuna"] = pd.to_numeric(c, errors="coerce")          # 'SD'/'nan' -> NaN

    # Tipo de vía: 'SD' -> NaN (sin dato)
    h["tipo_via"] = h["tipo_de_via_siniestro"].replace("SD", np.nan)

    # Fecha y variables temporales
    h["fecha"] = pd.to_datetime(h["fecha_siniestro"], errors="coerce")
    h["dia_semana"] = h["fecha"].dt.dayofweek
    h["fin_de_semana"] = h["dia_semana"].isin([5, 6]).astype(int)
    h["hora"] = pd.to_numeric(h["rango_horario"], errors="coerce")
    h["anio"] = h["fecha"].dt.year
    h["mes"] = h["fecha"].dt.month

    # Franja horaria (útil para el modelo y la visualización)
    h["franja"] = pd.cut(h["hora"], bins=[-1, 5, 11, 17, 23],
                         labels=["Madrugada", "Mañana", "Tarde", "Noche"])

    # Gravedad: variable objetivo
    h["gravedad"] = h["gravedad_siniestro"].str.upper()
    h["grave_o_mortal"] = h["gravedad"].isin(["GRAVE", "MORTAL"]).astype(int)

    # Categorías 'SD' (sin dato) -> NaN para no confundirlas con una clase real
    # (y se unifican mayúsculas: hay 'moto', 'camion', etc. en minúscula)
    for col in ["modo_desplazamiento_victima", "contraparte_siniestro"]:
        h[col] = h[col].astype("string").str.strip().str.upper().replace("SD", np.nan)
        h[col] = h[col].astype(object).where(h[col].notna(), np.nan)

    # Coordenadas: vienen con texto ('SD', '.') -> numérico; luego se descartan
    # las que caen fuera de CABA (errores de geocodificación)
    for col in ["latitud_siniestro", "longitud_siniestro"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")
    fuera = ~h["latitud_siniestro"].between(-34.75, -34.50) | \
            ~h["longitud_siniestro"].between(-58.55, -58.33)
    h.loc[fuera, ["latitud_siniestro", "longitud_siniestro"]] = np.nan
    return h


def limpiar_victimas(ruta_csv: str) -> pd.DataFrame:
    """Lee el CSV (separador ';') y unifica mayúsculas / valores 'SD'."""
    v = pd.read_csv(ruta_csv, sep=";", low_memory=False)
    v = v.dropna(subset=["id_siniestro"]).copy()             # 4 filas vacías

    for col in ["modo_desplazamiento_victima", "sexo_victima",
                "GRAVEdad_victima", "rol_victima"]:
        v[col] = v[col].astype(str).str.strip().str.upper()
    v = v.rename(columns={"GRAVEdad_victima": "gravedad_victima"})

    v["edad"] = pd.to_numeric(v["edad_victima"], errors="coerce")   # 'SD' -> NaN
    v.loc[(v["edad"] < 0) | (v["edad"] > 105), "edad"] = np.nan
    v["sexo"] = v["sexo_victima"].replace("SD", np.nan)
    v["modo"] = v["modo_desplazamiento_victima"].replace("SD", np.nan)
    v["anio"] = pd.to_numeric(v["anio_siniestro"], errors="coerce").astype("Int64")
    v["fecha"] = pd.to_datetime(v["fecha_siniestro"], errors="coerce")
    v["mortal"] = (v["gravedad_victima"] == "MORTAL").astype(int)
    return v


# 2. INDICADORES
def tabla_por_modo(v: pd.DataFrame) -> pd.DataFrame:
    """Víctimas por modo de desplazamiento y % de graves/mortales."""
    t = v.dropna(subset=["modo"]).groupby("modo").agg(
        victimas=("mortal", "size"),
        mortales=("mortal", "sum"),
        graves=("gravedad_victima", lambda s: (s == "GRAVE").sum()))
    t["pct_grave_mortal"] = (t["mortales"] + t["graves"]) / t["victimas"] * 100
    t["pct_mortal"] = t["mortales"] / t["victimas"] * 100
    return t.sort_values("victimas", ascending=False)


def tabla_por_comuna(h: pd.DataFrame) -> pd.DataFrame:
    t = h.dropna(subset=["comuna"]).groupby("comuna").agg(
        siniestros=("id_siniestro", "size"),
        victimas=("numero_total_de_victimas", "sum"),
        mortales=("numero_victimas_mortal_siniestro", "sum"))
    t.index = t.index.astype(int)
    return t


def test_hipotesis_moto(v: pd.DataFrame):
    """
    H1: la proporción de víctimas graves o mortales es mayor entre motociclistas
    que entre ocupantes de autos. Test chi-cuadrado + razón de proporciones.
    """
    from scipy.stats import chi2_contingency
    d = v[v["modo"].isin(["MOTO", "AUTO"])].copy()
    d["gm"] = d["gravedad_victima"].isin(["GRAVE", "MORTAL"])
    ct = pd.crosstab(d["modo"], d["gm"])
    chi2, p, _, _ = chi2_contingency(ct)
    p_moto = ct.loc["MOTO", True] / ct.loc["MOTO"].sum()
    p_auto = ct.loc["AUTO", True] / ct.loc["AUTO"].sum()
    return {"p_moto": p_moto, "p_auto": p_auto,
            "razon": p_moto / p_auto, "chi2": chi2, "p_valor": p, "tabla": ct}


# 3. MODELO PREDICTIVO
FEATURES_CAT = ["modo_desplazamiento_victima", "contraparte_siniestro",
                "tipo_via", "franja"]
FEATURES_NUM = ["hora", "dia_semana", "fin_de_semana", "comuna", "mes"]


def entrenar_modelo(h: pd.DataFrame, seed: int = 42):
    """
    Random Forest que estima la probabilidad de que un siniestro tenga
    víctimas GRAVES o MORTALES a partir de variables conocidas al momento
    del hecho (no usa la cantidad de víctimas para evitar fuga de datos).
    """
    d = h.copy()
    X = d[FEATURES_CAT + FEATURES_NUM]
    X[FEATURES_CAT] = X[FEATURES_CAT].astype(object).fillna("SD")
    X[FEATURES_NUM] = X[FEATURES_NUM].apply(pd.to_numeric, errors="coerce")
    X = X.fillna({c: -1 for c in FEATURES_NUM})
    y = d["grave_o_mortal"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
                              FEATURES_CAT)], remainder="passthrough")
    modelo = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_depth=14,
            class_weight="balanced", n_jobs=-1, random_state=seed))])
    modelo.fit(X_tr, y_tr)

    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metricas = {
        "auc": roc_auc_score(y_te, proba),
        "recall_grave": recall_score(y_te, pred),
        "reporte": classification_report(y_te, pred, output_dict=True),
        "matriz": confusion_matrix(y_te, pred),
        "base_rate": float(y.mean()),
    }
    # Importancias agregadas por variable original
    nombres = modelo["pre"].get_feature_names_out()
    imp = pd.Series(modelo["rf"].feature_importances_, index=nombres)
    agg = {}
    for n, val in imp.items():
        base = n.split("__", 1)[1]
        for c in FEATURES_CAT:
            if base.startswith(c):
                base = c
        agg[base] = agg.get(base, 0) + val
    metricas["importancias"] = pd.Series(agg).sort_values(ascending=False)
    return modelo, metricas


def predecir(modelo, modo, contraparte, tipo_via, hora, dia_semana, comuna, mes,
             prior=0.0554):
    """
    Devuelve la probabilidad (0-1) de siniestro grave o mortal.

    El modelo se entrena con class_weight="balanced" (para detectar mejor los
    casos graves, que son pocos), y eso infla las probabilidades: es como si
    los graves fueran el 50 % de los casos. Para devolver una probabilidad
    comparable con la frecuencia real (prior = % de graves en los datos),
    se corrige el desplazamiento de la tasa base sobre las chances (odds).
    """
    franja = ("Madrugada" if hora <= 5 else "Mañana" if hora <= 11
              else "Tarde" if hora <= 17 else "Noche")
    fila = pd.DataFrame([{
        "modo_desplazamiento_victima": modo, "contraparte_siniestro": contraparte,
        "tipo_via": tipo_via, "franja": franja, "hora": hora,
        "dia_semana": dia_semana, "fin_de_semana": int(dia_semana >= 5),
        "comuna": comuna, "mes": mes}])
    p = float(modelo.predict_proba(fila)[:, 1][0])
    odds = p / max(1 - p, 1e-9) * (prior / (1 - prior))   # corrección de tasa base
    return odds / (1 + odds)