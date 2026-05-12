"""
╔══════════════════════════════════════════════════════════════╗
║           CGM VALIDATOR  —  Version 1.1                      ║
║   Kontinuierliche Glukosemessung: Qualitätsanalyse & Report  ║
╚══════════════════════════════════════════════════════════════╝

Starten:
    streamlit run cgm_validator.py

Benötigte Pakete (einmalig):
    pip install streamlit pandas numpy matplotlib scipy reportlab
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from scipy import stats
from io import BytesIO
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  DESIGN SYSTEM — Helles, professionelles Medical-Theme
# ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#F7F9FC",   # Sehr helles Blaugrau
    "surface":   "#FFFFFF",   # Weiß
    "border":    "#DDE3ED",   # Helles Grau-Blau
    "primary":   "#0071BC",   # Medizinisches Blau
    "primary_lt":"#E8F2FB",   # Helles Blau (Hintergrund)
    "success":   "#1A9E5C",   # Grün
    "warning":   "#D97706",   # Amber
    "danger":    "#DC2626",   # Rot
    "text":      "#1A202C",   # Fast-Schwarz
    "muted":     "#5A6478",   # Mittelgrau
    "glucose":   "#0071BC",   # Blau für Glukose-Linie
    "flag":      "#DC2626",
    "plateau":   "#D97706",
    "gap":       "#7C3AED",
}

# ─────────────────────────────────────────────────────────────
#  SENSOR-KONFIGURATION
#  Liegedauer und Warmup-Phase je Sensor-Typ
# ─────────────────────────────────────────────────────────────
SENSOR_CONFIG = {
    # source_name (Teilstring)  →  (max_wear_days, warmup_h, gap_thresh_min)
    "Glooko":         (15, 1.0,  60),   # FreeStyle Libre 3: 15 Tage, 1h Warmup, 60 min Lücke
    "Abbott":         (15, 1.0,  60),   # LibreView direkt
    "Dexcom":         (10, 2.0, 120),   # G6/G7: 10 Tage, 2h Warmup
    "Medtronic":      ( 7, 2.0, 120),   # Guardian: 7 Tage
    "Nightscout":     (14, 1.0, 180),   # variabel, konservative Lücke
}

def get_sensor_config(source_name):
    """Gibt (max_wear_days, warmup_h, gap_thresh_min) für eine Quelle zurück."""
    for key, cfg in SENSOR_CONFIG.items():
        if key.lower() in source_name.lower():
            return cfg
    return (14, 2.0, 120)   # Fallback


st.set_page_config(
    page_title="CGM Validator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS — Professionelles Light-Theme
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
  }}
  .stApp {{ background-color: {COLORS['bg']}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background-color: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
  }}
  section[data-testid="stSidebar"] * {{
    color: {COLORS['text']} !important;
  }}

  /* Metriken */
  [data-testid="metric-container"] {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  [data-testid="metric-container"] label {{
    color: {COLORS['muted']} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {COLORS['text']} !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['surface']};
    border-radius: 8px;
    padding: 4px;
    border: 1px solid {COLORS['border']};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {COLORS['muted']};
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border-radius: 6px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {COLORS['primary']} !important;
    color: #fff !important;
  }}

  /* Buttons */
  .stDownloadButton button, .stButton button {{
    background: {COLORS['primary']};
    color: #fff;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    letter-spacing: 0.03em;
  }}
  .stDownloadButton button:hover, .stButton button:hover {{
    background: #005a96;
    color: #fff;
  }}

  /* File uploader */
  [data-testid="stFileUploader"] {{
    background: {COLORS['surface']};
    border: 2px dashed {COLORS['border']};
    border-radius: 10px;
    padding: 8px;
  }}

  /* Dataframe */
  .stDataFrame {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  /* Ueberschriften */
  h1 {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.02em;
    color: {COLORS['text']};
  }}
  h2, h3 {{ color: {COLORS['text']}; font-weight: 600; }}

  /* Info-Boxen */
  .info-card {{
    background: {COLORS['primary_lt']};
    border: 1px solid #B3D4EF;
    border-left: 4px solid {COLORS['primary']};
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: {COLORS['text']};
  }}
  .warn-card {{
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid {COLORS['warning']};
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: {COLORS['text']};
  }}
  .danger-card {{
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid {COLORS['danger']};
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: {COLORS['text']};
  }}

  /* Score Badge */
  .score-badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    padding: 8px 24px;
    border-radius: 12px;
    margin: 8px 0;
  }}
  .divider {{
    border: none;
    border-top: 1px solid {COLORS['border']};
    margin: 16px 0;
  }}

  /* Code inline */
  code {{
    background: #EEF2F7;
    color: #1A202C;
    padding: 1px 5px;
    border-radius: 4px;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CSV-PARSER: Automatische Header-Erkennung
#  Validiert gegen echte Exportdateien:
#    - Glooko (FreeStyle Libre 3, 1-min)
#    - Abbott LibreView direkt (FreeStyle Libre 3, 5-min, Komma-sep)
#    - Dexcom Clarity / G5/G6 App (5-min, Semikolon-sep, EGV-Filter)
#    - Medtronic CareLink 780G (5-min, Semikolon-sep, Datum+Zeit getrennt)
# ─────────────────────────────────────────────────────────────

def _clean_colname(c):
    """Entfernt BOM, Anführungszeichen, Leerzeichen."""
    return c.strip().lstrip('\ufeff').strip('"').strip()


def _read_raw(uploaded_file):
    """Liest Datei als Text, erkennt Encoding automatisch.
    Robust gegen Streamlit UploadedFile (SpooledTemporaryFile) und BytesIO.
    """
    raw = None
    # Versuch 1: getvalue() — funktioniert bei BytesIO
    try:
        raw = uploaded_file.getvalue()
    except AttributeError:
        pass
    # Versuch 2: seek(0) + read() — funktioniert bei SpooledTemporaryFile
    if not raw:
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
        except Exception:
            pass
    # Versuch 3: read() ohne seek — letzter Ausweg
    if not raw:
        try:
            raw = uploaded_file.read()
        except Exception:
            return ""
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _detect_sep(line):
    return ";" if line.count(";") >= line.count(",") else ","


def _parse_datetime_robust(series, fmts, dayfirst=True):
    """Versucht mehrere Datumsformate, gibt geparste Series oder None zurück."""
    for fmt in fmts:
        try:
            result = pd.to_datetime(series, format=fmt)
            if result.notna().sum() > 0:
                return result
        except Exception:
            continue
    try:
        result = pd.to_datetime(series, dayfirst=dayfirst, infer_datetime_format=True)
        if result.notna().sum() > 0:
            return result
    except Exception:
        pass
    return None


def parse_cgm_csv(uploaded_file):
    """
    Universeller CGM-Parser. Gibt zurück:
      (df, dt_col_orig, gl_col_orig, source_name, error_str)
    df hat Spalten: datetime (datetime64), glucose (float)
    """
    from io import StringIO

    # Pointer zurücksetzen — Streamlit UploadedFile kann bereits gelesen worden sein
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    content = _read_raw(uploaded_file)
    lines = [l for l in content.splitlines() if l.strip()]

    # ── Quellerkennung anhand des Rohinhalts ──────────────────

    content_lower = content.lower()

    # ══ MEDTRONIC CARELINK ════════════════════════════════════
    # Merkmale: Semikolon-sep, "SensorGlucose" Spalte im Header,
    #           Datum und Zeit in GETRENNTEN Spalten (Date / Time),
    #           6 Meta-Zeilen, Format: 2021/08/24 + 10:00:47
    #
    # ⚠ MEDTRONIC-FALLE: In Closed-Loop-Exporten (780G/670G) ist die
    #   SensorGlucose-Spalte leer. Der aktuelle CGM-Wert wird stattdessen
    #   als "Bolus Number" bei jedem Closed-Loop-Mikro-Bolus mitgespeichert.
    #   → Strategie: erst SensorGlucose versuchen, dann Bolus-Number-Fallback.
    if "sensorglucose" in content_lower or "sensor glucose" in content_lower.replace(";", " "):
        # Header-Zeile finden (enthält "Index;Date;Time")
        header_idx = 0
        for i, line in enumerate(lines[:20]):
            cl = line.lower()
            if "date" in cl and "time" in cl and ("index" in cl or "sensorglucose" in cl.replace(" ","")):
                header_idx = i
                break

        sep = _detect_sep(lines[header_idx])
        df_raw = pd.read_csv(StringIO(content), skiprows=header_idx, sep=sep,
                              on_bad_lines="skip", dtype=str, quotechar='"')
        df_raw.columns = [_clean_colname(c) for c in df_raw.columns]

        # Datum + Zeit zusammenführen
        date_col = next((c for c in df_raw.columns if c.lower() == "date"), None)
        time_col = next((c for c in df_raw.columns if c.lower() == "time"), None)
        gl_col   = next((c for c in df_raw.columns if "sensorglucose" in c.lower().replace(" ","")), None)

        if date_col and time_col:
            df_raw["_datetime_str"] = df_raw[date_col].str.strip() + " " + df_raw[time_col].str.strip()
            dt_fmts = ["%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S",
                       "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M"]

            # ── Strategie 1: SensorGlucose-Spalte (Standard-Export)
            if gl_col:
                df_sg = df_raw[["_datetime_str", gl_col]].copy()
                df_sg.columns = ["datetime", "glucose"]
                df_sg = df_sg[df_sg["glucose"].notna() & df_sg["glucose"].str.strip().ne("")]
                df_sg["glucose"] = df_sg["glucose"].str.replace(",", ".").str.extract(r"([\d.]+)")[0]
                df_sg["glucose"] = pd.to_numeric(df_sg["glucose"], errors="coerce")
                df_sg = df_sg.dropna(subset=["glucose"])
                df_sg = df_sg[(df_sg["glucose"] >= 20) & (df_sg["glucose"] <= 500)]

                if len(df_sg) >= 10:
                    dt_parsed = _parse_datetime_robust(df_sg["datetime"], dt_fmts)
                    if dt_parsed is not None:
                        df_sg["datetime"] = dt_parsed
                        df_sg = df_sg.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                        return df_sg, f"{date_col} + {time_col}", gl_col, "Medtronic CareLink", None

            # ── Strategie 2: Closed-Loop-Fallback — CGM-Wert steckt in "Bolus Number"
            #    Kennzeichen: CLOSED_LOOP_*-Zeilen, 5-min-Intervall, Wert 40–400
            bolus_src_col = next((c for c in df_raw.columns
                                  if "bolus source" in c.lower()), None)
            bolus_num_col = next((c for c in df_raw.columns
                                  if c.lower() == "bolus number"), None)

            if bolus_src_col and bolus_num_col:
                cl_rows = df_raw[
                    df_raw[bolus_src_col].str.contains("CLOSED_LOOP", na=False) &
                    df_raw[bolus_num_col].notna() &
                    df_raw[bolus_num_col].str.strip().ne("")
                ].copy()

                cl_rows["glucose"] = pd.to_numeric(
                    cl_rows[bolus_num_col].str.replace(",", "."), errors="coerce"
                )
                cl_rows = cl_rows[
                    (cl_rows["glucose"] >= 40) & (cl_rows["glucose"] <= 400)
                ].copy()

                if len(cl_rows) >= 10:
                    df_cl = cl_rows[["_datetime_str", "glucose"]].copy()
                    df_cl.columns = ["datetime", "glucose"]
                    dt_parsed = _parse_datetime_robust(df_cl["datetime"], dt_fmts)
                    if dt_parsed is not None:
                        df_cl["datetime"] = dt_parsed
                        df_cl = (df_cl.dropna(subset=["datetime"])
                                      .sort_values("datetime")
                                      .drop_duplicates(subset=["datetime"])
                                      .reset_index(drop=True))
                        # Validierung: Median-Intervall muss plausibel sein (3–7 min)
                        med_interval = (df_cl["datetime"].diff()
                                        .dt.total_seconds().dropna().median() / 60)
                        if 2 <= med_interval <= 10:
                            return (df_cl, f"{date_col} + {time_col}",
                                    f"{bolus_num_col} (Closed-Loop CGM-Referenz)",
                                    "Medtronic CareLink (Closed-Loop)", None)

            # ── Kein CGM gefunden
            return None, None, None, None, (
                "Medtronic CareLink erkannt, aber keine CGM-Messwerte gefunden.\n\n"
                "Diese Datei enthält nur Pumpen-Aktivitätsdaten ohne Sensor-Glukosewerte.\n"
                "Export-Tipp: CareLink → Berichte → Gerätedaten → "
                "Häkchen bei 'Sensordaten' setzen."
            )

        return None, None, None, None, "Medtronic: Spalten Date/Time nicht gefunden."

    # ══ DEXCOM CLARITY / G5 / G6 APP ═════════════════════════
    # Merkmale: Spalte "Ereignisart" / "Ereignis-Unterart",
    #           nur Zeilen mit Ereignisart == "EGV" enthalten CGM-Werte,
    #           Zeitstempel: ISO 8601 (2026-02-20T00:03:22),
    #           Semikolon-sep (deutsch), Glukosespalte: "Glukosewert (mg/dL)" oder "Glukosewert"
    if "ereignisart" in content_lower or "egv" in content_lower:
        # Header steht auf Zeile 0 (nach BOM)
        sep = _detect_sep(lines[0])
        df_raw = pd.read_csv(StringIO(content), sep=sep, on_bad_lines="skip",
                              dtype=str, quotechar='"')
        df_raw.columns = [_clean_colname(c) for c in df_raw.columns]

        # EGV-Zeilen filtern
        event_col = next((c for c in df_raw.columns if "ereignisart" in c.lower()), None)
        if event_col:
            df_raw = df_raw[df_raw[event_col].str.strip().str.upper() == "EGV"].copy()

        # Spalten finden
        dt_col = next((c for c in df_raw.columns
                       if "zeitstempel" in c.lower() or "timestamp" in c.lower()), None)
        gl_col = next((c for c in df_raw.columns
                       if "glukosewert" in c.lower() or "glucose value" in c.lower()
                       or (c.lower().startswith("glukosewert") or c.lower() == "glukosewert")), None)

        if dt_col and gl_col:
            df = df_raw[[dt_col, gl_col]].copy()
            df.columns = ["datetime", "glucose"]
            df = df[df["glucose"].str.strip().ne("") & df["glucose"].notna()]
            df["glucose"] = df["glucose"].str.replace(",", ".").str.extract(r"([\d.]+)")[0]
            df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
            df = df.dropna(subset=["glucose"])
            df = df[df["glucose"] > 0]
            dt_parsed = _parse_datetime_robust(
                df["datetime"],
                ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                 "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"]
            )
            if dt_parsed is not None:
                df["datetime"] = dt_parsed
                df = df.dropna(subset=["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                return df, dt_col, gl_col, "Dexcom Clarity", None

        return None, None, None, None, "Dexcom: EGV-Zeilen oder Glukosespalte nicht gefunden."

    # ══ ABBOTT LIBREVIEW (direkt) ═════════════════════════════
    # Merkmale: "Gerätezeitstempel" Spalte, "Aufzeichnungstyp" Spalte,
    #           nur Typ 0 = historische CGM-Werte,
    #           Komma-sep, Datum: DD-MM-YYYY HH:MM
    #           Glukosespalte: "Glukosewert-Verlauf mg/dL"
    if "gerätezeitstempel" in content_lower or "aufzeichnungstyp" in content_lower:
        # Header-Zeile finden (Zeile 2, nach 2 Metazeilen)
        header_idx = 0
        for i, line in enumerate(lines[:10]):
            cl = line.lower()
            if "gerätezeitstempel" in cl or "device timestamp" in cl:
                header_idx = i
                break

        sep = _detect_sep(lines[header_idx])
        df_raw = pd.read_csv(StringIO(content), skiprows=header_idx, sep=sep,
                              on_bad_lines="skip", dtype=str, quotechar='"')
        df_raw.columns = [_clean_colname(c) for c in df_raw.columns]

        # Aufzeichnungstyp 0 = historische Messung (CGM)
        type_col = next((c for c in df_raw.columns if "aufzeichnungstyp" in c.lower()
                         or "record type" in c.lower()), None)
        if type_col:
            df_raw = df_raw[df_raw[type_col].str.strip() == "0"].copy()

        dt_col = next((c for c in df_raw.columns
                       if "gerätezeitstempel" in c.lower() or "device timestamp" in c.lower()), None)
        gl_col = next((c for c in df_raw.columns
                       if "verlauf" in c.lower() or "historic" in c.lower()), None)
        # Fallback: erste Glukosespalte
        if gl_col is None:
            gl_col = next((c for c in df_raw.columns
                           if "glukose" in c.lower() or "glucose" in c.lower()), None)

        if dt_col and gl_col:
            df = df_raw[[dt_col, gl_col]].copy()
            df.columns = ["datetime", "glucose"]
            df = df[df["glucose"].str.strip().ne("") & df["glucose"].notna()]
            df["glucose"] = df["glucose"].str.replace(",", ".").str.extract(r"([\d.]+)")[0]
            df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
            df = df.dropna(subset=["glucose"])
            df = df[df["glucose"] > 0]
            dt_parsed = _parse_datetime_robust(
                df["datetime"],
                ["%d-%m-%Y %H:%M", "%d.%m.%Y %H:%M", "%m/%d/%Y %H:%M",
                 "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M"]
            )
            if dt_parsed is not None:
                df["datetime"] = dt_parsed
                df = df.dropna(subset=["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                return df, dt_col, gl_col, "Abbott LibreView", None

        return None, None, None, None, "LibreView: Gerätezeitstempel oder Glukosspalte nicht gefunden."

    # ══ GLOOKO / FREESTYLE LIBRE (1-min, deutsch) ════════════
    # Merkmale: "Zeitstempel" Spalte, "CGM-Glukosewert (mg/dl)" Spalte,
    #           erste Zeile = Metadaten (Name:xxx, Datumsbereich:xxx),
    #           Komma-sep, absteigend sortiert
    if "zeitstempel" in content_lower and "cgm" in content_lower:
        header_idx = 0
        for i, line in enumerate(lines[:10]):
            cl = line.lower()
            if "zeitstempel" in cl:
                header_idx = i
                break

        sep = _detect_sep(lines[header_idx])
        df_raw = pd.read_csv(StringIO(content), skiprows=header_idx, sep=sep,
                              on_bad_lines="skip", dtype=str, quotechar='"')
        df_raw.columns = [_clean_colname(c) for c in df_raw.columns]

        dt_col = next((c for c in df_raw.columns if "zeitstempel" in c.lower()), None)
        gl_col = next((c for c in df_raw.columns
                       if "glukose" in c.lower() or "glucose" in c.lower()), None)

        if dt_col and gl_col:
            df = df_raw[[dt_col, gl_col]].copy()
            df.columns = ["datetime", "glucose"]
            df = df[df["glucose"].str.strip().ne("") & df["glucose"].notna()]
            df["glucose"] = df["glucose"].str.replace(",", ".").str.extract(r"([\d.]+)")[0]
            df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
            df = df.dropna(subset=["glucose"])
            df = df[df["glucose"] > 0]
            dt_parsed = _parse_datetime_robust(
                df["datetime"],
                ["%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M",
                 "%d-%m-%Y %H:%M"]
            )
            if dt_parsed is not None:
                df["datetime"] = dt_parsed
                df = df.dropna(subset=["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                return df, dt_col, gl_col, "Glooko / FreeStyle Libre", None

        return None, None, None, None, "Glooko: Zeitstempel oder Glukosspalte nicht gefunden."

    # ══ NIGHTSCOUT ════════════════════════════════════════════
    # Merkmale: Spalten "_id", "sgv", "dateString", type="sgv"
    # Zeitstempel: ISO8601, gemischt (mit/ohne Millisekunden, UTC oder Offset)
    # Quelle: Nightscout-Datenbankexport (entries.csv / heroku-Export)
    if "_id" in content_lower and "sgv" in content_lower and "datestring" in content_lower:
        try:
            df_raw = pd.read_csv(StringIO(content), dtype=str, on_bad_lines="skip")
            df_raw.columns = [c.strip() for c in df_raw.columns]

            # Nur SGV-Einträge (Sensor Glucose Values), keine pumpdata etc.
            if "type" in df_raw.columns:
                df_raw = df_raw[df_raw["type"].str.strip().str.lower() == "sgv"]

            # Glukosespalte: bevorzuge "sgv", fallback "glucose"
            gl_col = "sgv" if "sgv" in df_raw.columns else "glucose"
            dt_col = "dateString"

            if dt_col not in df_raw.columns or gl_col not in df_raw.columns:
                return None, None, None, None, "Nightscout: Spalten dateString/sgv nicht gefunden."

            df = df_raw[[dt_col, gl_col]].copy()
            df.columns = ["datetime", "glucose"]
            df = df[df["glucose"].notna() & df["glucose"].str.strip().ne("")]
            df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
            df = df.dropna(subset=["glucose"])
            df = df[df["glucose"] > 0]

            # ISO8601 mit gemischten Formaten ("+01:00" und ".000Z")
            df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
            df["datetime"] = df["datetime"].dt.tz_localize(None)  # Timezone entfernen
            df = df.dropna(subset=["datetime"])

            # Duplikate entfernen (Nightscout kann mehrfache Einträge pro Zeitstempel haben)
            df = df.drop_duplicates(subset=["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            if len(df) < 10:
                return None, None, None, None, "Nightscout: Zu wenige SGV-Einträge nach Filterung."

            return df, dt_col, gl_col, "Nightscout", None

        except Exception as e:
            return None, None, None, None, f"Nightscout-Parser Fehler: {e}"

    # ══ GENERISCHER FALLBACK ══════════════════════════════════
    # Findet Header-Zeile heuristisch und versucht Spalten zu erkennen
    header_idx = 0
    for i, line in enumerate(lines[:20]):
        cl = line.lower()
        parts = [p.strip() for p in re.split(r'[,;]', line) if p.strip()]
        if len(parts) >= 2 and any(kw in cl for kw in
                ["zeit", "time", "date", "datum", "glucose", "glukose", "sensor", "timestamp"]):
            header_idx = i
            break

    sep = _detect_sep(lines[header_idx] if header_idx < len(lines) else lines[0])
    df_raw = pd.read_csv(StringIO(content), skiprows=header_idx, sep=sep,
                         on_bad_lines="skip", dtype=str, quotechar='"')
    df_raw.columns = [_clean_colname(c) for c in df_raw.columns]

    dt_col = next((c for c in df_raw.columns
                   if any(k in c.lower() for k in ["zeit", "time", "date", "datum", "stamp"])), None)
    gl_col = next((c for c in df_raw.columns
                   if any(k in c.lower() for k in ["glukose", "glucose", "sensor", "mg", "mmol"])), None)

    if dt_col is None or gl_col is None:
        return None, None, None, None, (
            f"Format nicht erkannt. Gefundene Spalten: {list(df_raw.columns)}\n"
            "Bitte stellen Sie sicher, dass die Datei eine DateTime-Spalte und eine Glukosespalte enthält."
        )

    df = df_raw[[dt_col, gl_col]].copy()
    df.columns = ["datetime", "glucose"]
    df = df[df["glucose"].str.strip().ne("") & df["glucose"].notna()]
    df["glucose"] = df["glucose"].str.replace(",", ".").str.extract(r"([\d.]+)")[0]
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df = df.dropna(subset=["glucose"])
    df = df[df["glucose"] > 0]
    dt_parsed = _parse_datetime_robust(df["datetime"], [], dayfirst=True)
    if dt_parsed is None:
        return None, None, None, None, "DateTime konnte nicht geparst werden."
    df["datetime"] = dt_parsed
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df, dt_col, gl_col, "Unbekannte Quelle (Heuristik)", None


# ─────────────────────────────────────────────────────────────
#  ANALYSE-MODULE
# ─────────────────────────────────────────────────────────────

def detect_interval(df):
    """Erkennt das Messintervall (1 min oder 5 min).
    Verwendet den Modus (häufigster Wert) statt Median, da Nightscout-Daten
    nach Deduplizierung unregelmäßige Abstände haben können.
    """
    diffs = df["datetime"].diff().dt.total_seconds().dropna()
    # Runde auf 30-Sekunden-Raster um Jitter zu eliminieren
    diffs_rounded = (diffs / 30).round() * 30
    mode_sec = diffs_rounded.mode().iloc[0] if len(diffs_rounded) > 0 else 300
    if mode_sec <= 90:
        return 1
    elif mode_sec <= 360:
        return 5
    else:
        return round(mode_sec / 60)


def analyze_rate_of_change(df, interval_min):
    """Modul 1: Physiologisch implausible Änderungsraten."""
    threshold = 4.0 if interval_min == 1 else 20.0
    df = df.copy()
    df["delta"] = df["glucose"].diff().abs()
    df["roc_flag"] = df["delta"] > threshold
    return df, threshold


def analyze_gaps(df, interval_min):
    """Modul 2: Fehlende Messungen / Zeitlücken."""
    df = df.copy()
    expected_gap = timedelta(minutes=interval_min)
    df["time_diff"] = df["datetime"].diff()
    tolerance = timedelta(seconds=30) if interval_min == 1 else timedelta(minutes=2)
    df["gap_flag"] = df["time_diff"] > (expected_gap + tolerance)
    df["gap_minutes"] = df["time_diff"].dt.total_seconds() / 60
    gaps = df[df["gap_flag"]].copy()
    return df, gaps


def analyze_plateaus(df, interval_min, manual_min=None):
    """Modul 3: Signal-Sticking — kontextbewusster Plateau-Detektor.

    Zwei Kriterien (OR-Verknüpfung):
      A) Langer Stillstand:       ≥20 identische Werte bei 1-min  (= 20 min)
                                  ≥ 6 identische Werte bei 5-min  (= 30 min)
         → immer verdächtig, unabhängig vom Kontext

      B) Kontextuelles Einfrieren: ≥8 identische Werte (1-min) + Sprung >4 mg/dL
                                   ≥6 identische Werte (5-min) + Sprung >8 mg/dL
         → Sensor war blockiert, dann Nachholeffekt

    Hintergrund 1-min (FL3): Kurze identische Runs ohne Sprung sind meist
    Kalman-Filter-Glättung und keine Artefakte (~99% False Positives empirisch).

    Hintergrund 5-min (Dexcom/Nightscout): Bis zu 65% der Punkte liegen in
    Runs ≥2 identischer Werte — physiologisch normal bei ganzzahligen mg/dL-
    Werten und 5-min-Abstand. Daher strenge Schwellen: ≥6 Werte (30 min)
    UND Sprung >8 mg/dL. Empirisch: ~7 echte Ereignisse in 47.000 Punkten.
    """
    df = df.copy()
    df["plateau_flag"]   = False
    df["plateau_length"] = 0

    glucose = df["glucose"].values
    n = len(glucose)

    if manual_min is not None:
        min_short   = manual_min
        min_long    = manual_min
        # Kontext-Sprung bleibt intervall-abhängig auch im manuellen Modus
        jump_thresh = 4.0 if interval_min == 1 else 8.0
    elif interval_min == 1:
        min_short   = 8     # Kriterium B: kurz aber mit Kontext
        min_long    = 20    # Kriterium A: ≥20 min immer flaggen
        jump_thresh = 4.0
    else:
        # 5-min-Daten (Dexcom, Nightscout, Medtronic): Plateau deaktiviert.
        # Bei 5-min-Intervall sind identische Werte über 30 min physiologisch
        # normal (ganzzahlige mg/dL-Rundung + stabile Glukosephasen).
        # RoC und Kompression decken echte Artefakte zuverlässiger ab.
        return df

    i = 0
    while i < n:
        j = i + 1
        while j < n and glucose[j] == glucose[i]:
            j += 1
        run_len = j - i

        if run_len >= min_short:
            jump_before = abs(glucose[i]   - glucose[i - 1]) if i > 0 else 0.0
            jump_after  = abs(glucose[j]   - glucose[j - 1]) if j < n else 0.0
            has_context = (jump_before > jump_thresh) or (jump_after > jump_thresh)

            if run_len >= min_long or has_context:
                df.iloc[i:j, df.columns.get_loc("plateau_flag")]   = True
                df.iloc[i:j, df.columns.get_loc("plateau_length")] = run_len

        i = j

    return df


def analyze_statistical_outliers(df, window_min=30, interval_min=1):
    """Modul 4: Lokale statistische Ausreißer im gleitenden Fenster."""
    df = df.copy()
    window_size = max(5, window_min // interval_min)
    df["zscore_flag"] = False
    df["zscore"] = 0.0

    glucose = df["glucose"].values
    n = len(glucose)
    for i in range(n):
        lo = max(0, i - window_size // 2)
        hi = min(n, i + window_size // 2)
        window = glucose[lo:hi]
        if len(window) < 5:
            continue
        z = abs((glucose[i] - np.mean(window)) / (np.std(window) + 1e-6))
        df.iloc[i, df.columns.get_loc("zscore")] = z
        if z > 3.5:
            df.iloc[i, df.columns.get_loc("zscore_flag")] = True
    return df


def analyze_compression(df, interval_min):
    """
    Modul 5: Kompressionsartefakte — ereignisbasierter Algorithmus.

    Strategie:
      1. Lokale Minima (Tiefpunkte) suchen
      2. Davor: Abfall ≥ 15 mg/dL innerhalb 30 min?
      3. Danach: Erholung ≥ 15 mg/dL innerhalb 45 min?
      4. Erholungsgeschwindigkeit > 3 mg/dL/min
         (physiologische Gegenregulation braucht länger)
      5. Nahe Tiefpunkte (< 20 min) zu einem Ereignis zusammenführen

    Kalibriert auf reale 1-min FL3-Daten (cgm_data_1.csv):
    Typische Ereignisdauer: 10–70 min, Erholung explosiver als Abfall.
    """
    df = df.copy()
    df["compression_flag"] = False

    g = df["glucose"].values
    n = len(g)

    lb       = 30 // interval_min   # Lookback-Fenster
    lf       = 45 // interval_min   # Lookahead-Fenster
    win_loc  = max(5, 10 // interval_min)  # Lokales-Minimum-Fenster
    merge_gap = 20 // interval_min  # Zwei Tiefpunkte < 20 min → ein Ereignis

    drop_thr     = 15.0   # min. Abfall zum Tiefpunkt (mg/dL)
    rec_thr      = 15.0   # min. Erholung vom Tiefpunkt (mg/dL)
    min_rec_speed = 3.0   # min. Erholungsgeschwindigkeit (mg/dL/min)

    raw_troughs = []

    for i in range(lb, n - lf):
        # Lokales Minimum?
        lo = max(0, i - win_loc)
        hi = min(n, i + win_loc + 1)
        if g[i] != np.min(g[lo:hi]):
            continue
        # Abfall davor
        before = g[max(0, i - lb):i]
        if len(before) == 0 or np.max(before) - g[i] < drop_thr:
            continue
        # Erholung danach
        after = g[i:min(n, i + lf)]
        if len(after) < 3 or np.max(after) - g[i] < rec_thr:
            continue
        # Erholungsgeschwindigkeit
        rec_idx = int(np.argmax(after))
        max_rate = 0.0
        for j in range(min(rec_idx, lf - 1)):
            w = min(5 // max(interval_min, 1), rec_idx - j)
            if w > 0 and i + j + w < n:
                rate = (g[i + j + w] - g[i + j]) / w
                max_rate = max(max_rate, rate)
        if max_rate < min_rec_speed:
            continue
        raw_troughs.append(i)

    if not raw_troughs:
        return df

    # Tiefpunkte clustern: < merge_gap zusammenfassen
    clusters = []
    cluster = [raw_troughs[0]]
    for idx in raw_troughs[1:]:
        if idx - cluster[-1] <= merge_gap:
            cluster.append(idx)
        else:
            clusters.append(cluster)
            cluster = [idx]
    clusters.append(cluster)

    # Pro Cluster: Flaggen setzen
    for cluster in clusters:
        i = min(cluster, key=lambda x: g[x])  # tiefstes Minimum

        # Abfall-Beginn: rückwärts bis Wert deutlich über Tief
        drop_start = max(0, i - lb)
        for k in range(i - 1, max(0, i - lb) - 1, -1):
            if g[k] > g[i] + 10:
                drop_start = k
                break

        # Erholungs-Ende: vorwärts bis Wert wieder über Tief + rec_thr
        rec_end = min(n - 1, i + lf)
        for k in range(i + 1, min(n, i + lf)):
            if g[k] > g[i] + rec_thr:
                rec_end = min(n - 1, k + 5)
                break

        df.iloc[drop_start:rec_end + 1,
                df.columns.get_loc("compression_flag")] = True

    return df


def compute_quality_score(df, gaps_df, interval_min):
    """Berechnet einen Gesamt-Qualitätsscore (0–100)."""
    n = len(df)
    if n == 0:
        return 0

    n_roc    = df["roc_flag"].sum()
    n_gap    = len(gaps_df)
    n_plat   = df["plateau_flag"].sum()
    n_stat   = df["zscore_flag"].sum()
    n_comp   = df["compression_flag"].sum()

    # Abzüge in Prozentpunkten (0–100), kalibriert:
    # Maximaler Abzug wird bei ~5% Fehlerrate erreicht
    penalty = (
        min((n_roc  / n) *  800, 40) +   # RoC: max 40 Punkte bei 5% Fehlerrate
        min( n_gap  *  0.5,      15) +   # Lücken: je 0.5 Punkte, max 15
        min((n_plat / n) *  400, 20) +   # Plateaus: max 20 Punkte bei 5%
        min((n_stat / n) * 1500, 15) +   # Stat. Ausreißer: max 15 bei 1%
        min((n_comp / n) * 1000, 10) +   # Kompression: max 10 bei 1%
        0
    )
    score = max(0, min(100, 100 - penalty))
    return round(score, 1)


# ─────────────────────────────────────────────────────────────
#  VISUALISIERUNGEN
# ─────────────────────────────────────────────────────────────

def style_fig(fig):
    fig.patch.set_facecolor(COLORS["bg"])
    return fig


def plot_glucose_timeline(df):
    """Glukose-Zeitverlauf mit farblichen Markierungen."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor("#FAFBFD")

    # Zielbereich
    ax.axhspan(70, 180, alpha=0.07, color=COLORS["primary"], zorder=0)
    ax.axhline(70,  color=COLORS["warning"], linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axhline(180, color=COLORS["warning"], linewidth=0.9, linestyle="--", alpha=0.7)

    # Basislinie
    ax.plot(df["datetime"], df["glucose"], color=COLORS["glucose"],
            linewidth=1.1, alpha=0.9, zorder=2)

    # Flags überlagern
    flag_types = [
        ("roc_flag",         COLORS["flag"],    "RoC-Fehler",        80),
        ("plateau_flag",     COLORS["plateau"], "Plateau",           60),
        ("zscore_flag",      "#7C3AED",         "Stat. Ausreißer",   40),
        ("compression_flag", "#DB2777",         "Kompression",       20),
    ]
    for col, color, label, size in flag_types:
        if col in df.columns:
            flagged = df[df[col]]
            if len(flagged):
                ax.scatter(flagged["datetime"], flagged["glucose"],
                           c=color, s=size, zorder=5, label=label,
                           edgecolors="white", linewidths=0.4, alpha=0.95)

    ax.set_xlabel("Datum / Uhrzeit", color=COLORS["muted"], fontsize=9)
    ax.set_ylabel("Glukose (mg/dL)", color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax.legend(loc="upper right", fontsize=8, facecolor=COLORS["surface"],
              edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    ax.set_title("Glukose-Zeitverlauf mit Fehlmarkierungen",
                 color=COLORS["text"], fontsize=11, pad=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_poincare(df):
    """Poincaré-Plot: G(t) vs G(t+1)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor("#FAFBFD")

    g = df["glucose"].values
    x, y = g[:-1], g[1:]

    lims = [min(g.min(), 40), max(g.max(), 300)]
    ax.plot(lims, lims, color=COLORS["muted"], linewidth=0.8, linestyle="--", alpha=0.5)

    roc_flags = df["roc_flag"].values[1:]
    normal = ~roc_flags
    ax.scatter(x[normal], y[normal], c=COLORS["primary"], s=6, alpha=0.25, zorder=2)
    if roc_flags.sum() > 0:
        ax.scatter(x[roc_flags], y[roc_flags], c=COLORS["flag"], s=22,
                   alpha=0.9, zorder=4, label="RoC-Fehler",
                   edgecolors="white", linewidths=0.4)

    sd2 = np.std((y - x) / np.sqrt(2))
    sd1 = np.std((y + x) / np.sqrt(2))
    from matplotlib.patches import Ellipse
    center = (np.mean(x), np.mean(y))
    ellipse = Ellipse(xy=center, width=4 * sd1, height=4 * sd2,
                      angle=45, fill=False, edgecolor=COLORS["primary"],
                      linewidth=1.5, linestyle="-", alpha=0.8, zorder=3)
    ax.add_patch(ellipse)

    ax.set_xlabel("G(t)  [mg/dL]", color=COLORS["muted"], fontsize=9)
    ax.set_ylabel("G(t+1)  [mg/dL]", color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax.set_title("Poincaré-Plot", color=COLORS["text"],
                 fontsize=11, fontweight="bold", pad=12)

    textstr = f"SD1 = {sd1:.1f}\nSD2 = {sd2:.1f}\nSD2/SD1 = {sd2/sd1:.2f}"
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', color=COLORS["text"],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["surface"],
                      edgecolor=COLORS["border"], alpha=0.95))
    if roc_flags.sum() > 0:
        ax.legend(fontsize=8, facecolor=COLORS["surface"],
                  edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    plt.tight_layout()
    return fig


def plot_delta_distribution(df, threshold):
    """Histogramm der absoluten Änderungen."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor("#FAFBFD")

    deltas = df["delta"].dropna()
    ax.hist(deltas[deltas <= threshold * 3], bins=60, color=COLORS["primary"],
            alpha=0.6, edgecolor=COLORS["surface"])
    ax.axvline(threshold, color=COLORS["flag"], linewidth=1.8,
               linestyle="--", label=f"Schwelle: {threshold} mg/dL")

    ax.set_xlabel("Absoluter Δ-Glukose (mg/dL)", color=COLORS["muted"], fontsize=9)
    ax.set_ylabel("Häufigkeit", color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax.legend(fontsize=8, facecolor=COLORS["surface"],
              edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    ax.set_title("Verteilung der Änderungsraten", color=COLORS["text"],
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_zscore_timeline(df):
    """Z-Score Verlauf über Zeit."""
    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor("#FAFBFD")

    ax.plot(df["datetime"], df["zscore"], color=COLORS["muted"],
            linewidth=0.6, alpha=0.8)
    ax.axhline(3.5, color=COLORS["flag"], linewidth=1.2, linestyle="--",
               alpha=0.85, label="Schwelle: z = 3.5")

    flagged = df[df["zscore_flag"]]
    if len(flagged):
        ax.scatter(flagged["datetime"], flagged["zscore"], c=COLORS["flag"],
                   s=28, zorder=4, edgecolors="white", linewidths=0.4)

    ax.set_ylabel("Lokaler Z-Score", color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax.legend(fontsize=8, facecolor=COLORS["surface"],
              edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    ax.set_title("Statistischer Ausreißer-Score (lokaler Z-Score)",
                 color=COLORS["text"], fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig
    ax.legend(fontsize=8, facecolor=COLORS["surface"],
              edgecolor=COLORS["border"], labelcolor=COLORS["text"])
    ax.set_title("Statistischer Ausreißer-Score (lokaler Z-Score)",
                 color=COLORS["text"], fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_daily_quality(df):
    """Tages-Qualitätsübersicht: Fehlerrate pro Tag als Balkendiagramm."""
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    df["any_flag"] = (df["roc_flag"] | df["plateau_flag"] |
                      df["zscore_flag"] | df["compression_flag"])

    daily = df.groupby("date").agg(
        total=("glucose", "count"),
        flagged=("any_flag", "sum"),
        gaps=("gap_flag", "sum"),
    ).reset_index()
    daily["error_pct"] = 100 * daily["flagged"] / daily["total"].clip(lower=1)
    daily["availability"] = 100 * daily["total"] / daily["total"].max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    fig.patch.set_facecolor(COLORS["surface"])

    for ax in [ax1, ax2]:
        ax.set_facecolor("#FAFBFD")
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.tick_params(colors=COLORS["muted"], labelsize=8)

    # Farbe je nach Fehlerrate
    bar_colors = []
    for pct in daily["error_pct"]:
        if pct < 1:
            bar_colors.append(COLORS["primary"])
        elif pct < 5:
            bar_colors.append(COLORS["warning"])
        else:
            bar_colors.append(COLORS["danger"])

    x = np.arange(len(daily))
    ax1.bar(x, daily["error_pct"], color=bar_colors, width=0.7, zorder=2)
    ax1.axhline(1, color=COLORS["muted"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Flagged (%)", color=COLORS["muted"], fontsize=9)
    ax1.set_title("Tagesweise Datenqualität", color=COLORS["text"],
                  fontsize=11, fontweight="bold", pad=12)

    # Legende manuell
    legend_patches = [
        mpatches.Patch(color=COLORS["primary"], label="< 1% → Gut"),
        mpatches.Patch(color=COLORS["warning"], label="1–5% → Prüfen"),
        mpatches.Patch(color=COLORS["danger"],  label="> 5% → Kritisch"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, facecolor=COLORS["surface"],
               edgecolor=COLORS["border"], labelcolor=COLORS["text"], loc="upper right")

    # Verfügbarkeit
    ax2.bar(x, daily["availability"], color=COLORS["primary"], width=0.7,
            alpha=0.35, zorder=2)
    ax2.set_ylabel("Verfügb. (%)", color=COLORS["muted"], fontsize=8)
    ax2.set_ylim(0, 115)
    ax2.axhline(100, color=COLORS["muted"], linewidth=0.5, linestyle="--", alpha=0.4)

    # X-Achse: Datum
    if len(daily) <= 35:
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [str(d)[5:] for d in daily["date"]],
            rotation=45, ha="right", fontsize=7, color=COLORS["muted"]
        )
    else:
        step = max(1, len(daily) // 20)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels(
            [str(daily["date"].iloc[i])[5:] for i in range(0, len(daily), step)],
            rotation=45, ha="right", fontsize=7, color=COLORS["muted"]
        )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
#  SENSOR-ERKENNUNG & LIEGEZEIT-ANALYSE
# ─────────────────────────────────────────────────────────────

def detect_sensors(df, gap_thresh_min=60, max_wear_days=15):
    """
    Erkennt Sensor-Wechsel anhand von zwei Kriterien:

    1. Zeitlücke > gap_thresh_min  (primär — expliziter Warmup-Stopp)
    2. Laufzeit > max_wear_days    (sekundär — Failsafe wenn Lücke fehlt oder
       durch Deduplizierung von ZIP-Überlappungen verschwindet)

    Gibt Liste von dicts zurück:
        sensor_id, start, end, wear_days, n_measurements, split_reason
    """
    diffs = df["datetime"].diff().dt.total_seconds() / 60

    # Alle Bruchpunkte sammeln: Lücken + Zeitüberschreitungen
    break_indices = set()

    # Kriterium 1: explizite Zeitlücken
    for idx in df.index[diffs > gap_thresh_min]:
        break_indices.add(int(idx))

    # Kriterium 2: max. Liegedauer — innerhalb jedes lückenlosen Blocks prüfen
    # Segmente anhand der bereits gefundenen Lücken
    gap_breaks = sorted([0] + list(df.index[diffs > gap_thresh_min]) + [len(df)])
    for i in range(len(gap_breaks) - 1):
        s = gap_breaks[i]
        e = gap_breaks[i + 1]
        seg_times = df["datetime"].iloc[s:e]
        t0 = seg_times.iloc[0]
        # Innerhalb dieses lückenlosen Blocks: alle max_wear_days-Überschreitungen teilen
        for j in range(s, e):
            elapsed_days = (df["datetime"].iloc[j] - t0).total_seconds() / 86400
            if elapsed_days >= max_wear_days:
                # Suche den Punkt nach exakt max_wear_days als Schnittlinie
                break_indices.add(j)
                # Neuer t0 für den nächsten Sensor
                t0 = df["datetime"].iloc[j]

    break_list = sorted([0] + list(break_indices) + [len(df)])
    # Duplikate entfernen
    break_list = sorted(set(break_list))

    sensors = []
    sensor_id = 1
    for i in range(len(break_list) - 1):
        s = break_list[i]
        e = break_list[i + 1] - 1
        seg = df.iloc[s:e + 1]
        if len(seg) < 10:
            continue
        t_start  = seg["datetime"].iloc[0]
        t_end    = seg["datetime"].iloc[-1]
        wear_h   = (t_end - t_start).total_seconds() / 3600

        # Feststellen warum dieser Sensor begann
        if i > 0 and break_list[i] in set(df.index[diffs > gap_thresh_min]):
            reason = "Lücke"
        elif wear_h / 24 < max_wear_days - 0.5 and i > 0:
            reason = f"Max. Liegezeit ({max_wear_days}d)"
        else:
            reason = "Start"

        sensors.append({
            "sensor_id":    sensor_id,
            "start":        t_start,
            "end":          t_end,
            "wear_days":    wear_h / 24,
            "wear_hours":   wear_h,
            "n":            len(seg),
            "idx_start":    s,
            "idx_end":      e,
            "split_reason": reason,
        })
        sensor_id += 1

    return sensors


def analyze_wear_time(df, sensors, warmup_h=1.0):
    """
    Berechnet pro Sensor und Liegetag:
        - RoC-Fehlerrate
        - Kompressionsrate
        - Plateau-Rate
        - Verfügbarkeit
        - Nacht-Stabilitätswert (std 22–07 Uhr)
    Gibt DataFrame zurück.
    """
    rows = []
    for s in sensors:
        seg = df.iloc[s["idx_start"]:s["idx_end"] + 1].copy().reset_index(drop=True)
        t0  = s["start"]

        for day_num in range(1, int(s["wear_days"]) + 2):
            day_start = t0 + pd.Timedelta(hours=(day_num - 1) * 24)
            day_end   = t0 + pd.Timedelta(hours=day_num * 24)
            d = seg[(seg["datetime"] >= day_start) & (seg["datetime"] < day_end)]
            if len(d) < 10:
                continue

            is_warmup = day_num == 1   # erster Tag = Einlaufphase

            night = d[d["datetime"].dt.hour.isin(list(range(22, 24)) + list(range(0, 7)))]
            night_comp_pct = (100 * night["compression_flag"].sum() / len(night)
                              if len(night) >= 10 else float("nan"))

            rows.append({
                "sensor_id":      s["sensor_id"],
                "day":            day_num,
                "is_warmup":      is_warmup,
                "n":              len(d),
                "roc_pct":        100 * d["roc_flag"].sum()         / len(d),
                "comp_pct":       100 * d["compression_flag"].sum() / len(d),
                "plat_pct":       100 * d["plateau_flag"].sum()     / len(d),
                "night_comp_pct": night_comp_pct,
                "date_label":     day_start.strftime("%d.%m"),
            })
    return pd.DataFrame(rows)


def plot_wear_heatmap(wear_df, sensors):
    """Heatmap: Sensor × Liegetag, Farbe = Kompressionsrate."""
    C = COLORS
    n_sensors = len(sensors)
    max_day   = int(wear_df["day"].max())

    heat_h = max(5.0, n_sensors * 0.38)
    fig, ax_heat = plt.subplots(figsize=(17, heat_h), facecolor=C["bg"])
    ax_heat.set_facecolor("#FAFBFD")
    for spine in ax_heat.spines.values():
        spine.set_color(C["border"])
    ax_heat.tick_params(colors=C["muted"], labelsize=8)

    matrix  = np.full((n_sensors, max_day), np.nan)
    ylabels = []
    for row_i, s in enumerate(sensors):
        sub = wear_df[wear_df["sensor_id"] == s["sensor_id"]]
        for _, r in sub.iterrows():
            col_j = int(r["day"]) - 1
            if 0 <= col_j < max_day:
                matrix[row_i, col_j] = r["comp_pct"]
        ylabels.append(f"#{s['sensor_id']:02d}  {s['start'].strftime('%d.%m.%y')}")

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cgm_heat",
        [(0.0, "#E8F5E9"), (0.2, "#FFF9C4"), (0.5, "#FFB300"), (1.0, "#C62828")],
    )
    vmax = max(20.0, float(np.nanpercentile(matrix, 95)))
    im   = ax_heat.imshow(matrix, aspect="auto", cmap=cmap,
                          vmin=0, vmax=vmax, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Kompression (%)", fontsize=9, color=C["muted"])
    cbar.ax.tick_params(labelsize=8, colors=C["muted"])

    lbl_fs = max(5.0, min(8.5, 220 / max(n_sensors, 1)))
    ax_heat.set_yticks(np.arange(n_sensors))
    ax_heat.set_yticklabels(ylabels, fontsize=lbl_fs,
                            color=C["text"], fontfamily="monospace")
    for row_i in range(n_sensors):
        if row_i % 2 == 0:
            ax_heat.axhspan(row_i - 0.5, row_i + 0.5,
                            color="#F0F4F8", alpha=0.35, zorder=0)

    step   = max(1, max_day // 15)
    xticks = np.arange(0, max_day, step)
    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels([str(t + 1) for t in xticks], fontsize=8)
    ax_heat.set_xlabel("Liegetag", fontsize=9, color=C["muted"])
    ax_heat.set_title("Kompressionsrate nach Sensor und Liegetag (%)",
                      fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax_heat.axvline(0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
    for y in np.arange(0.5, n_sensors - 0.5):
        ax_heat.axhline(y, color=C["border"], linewidth=0.4, alpha=0.6)

    if n_sensors <= 25 and max_day <= 16:
        for row_i in range(n_sensors):
            for col_j in range(max_day):
                v = matrix[row_i, col_j]
                if not np.isnan(v):
                    txt_col = "white" if v > vmax * 0.55 else C["text"]
                    ax_heat.text(col_j, row_i, f"{v:.0f}",
                                 ha="center", va="center",
                                 fontsize=5.5, color=txt_col)
    plt.tight_layout()
    return fig


def plot_wear_boxplot(wear_df):
    """Boxplot: Streuung der Kompressionsrate pro Liegetag über alle Sensoren."""
    C = COLORS
    max_day = int(wear_df["day"].max())

    fig, ax = plt.subplots(figsize=(17, 5), facecolor=C["bg"])
    ax.set_facecolor("#FAFBFD")
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    ax.tick_params(colors=C["muted"], labelsize=9)

    box_data, box_positions = [], []
    for day in range(1, max_day + 1):
        vals = wear_df[wear_df["day"] == day]["comp_pct"].dropna().values
        if len(vals) >= 2:
            box_data.append(vals)
            box_positions.append(day)

    if box_data:
        ax.boxplot(box_data, positions=box_positions, widths=0.6,
                   patch_artist=True,
                   medianprops=dict(color=C["danger"], linewidth=2.5),
                   whiskerprops=dict(color=C["muted"], linewidth=1.2),
                   capprops=dict(color=C["muted"], linewidth=1.2),
                   flierprops=dict(marker="o", color=C["warning"],
                                   markersize=4, alpha=0.6),
                   boxprops=dict(facecolor=C["primary_lt"],
                                 color=C["primary"], linewidth=1.2))

        medians = [np.median(d) for d in box_data]
        ax.plot(box_positions, medians, "o-", color=C["primary"],
                linewidth=2, markersize=4, alpha=0.7, zorder=5)

        if len(box_positions) >= 3:
            z = np.polyfit(box_positions, medians, 1)
            p = np.poly1d(z)
            x_trend = np.array([box_positions[0], box_positions[-1]])
            direction = "↑" if z[0] > 0 else "↓"
            ax.plot(x_trend, p(x_trend), "--", color=C["danger"],
                    linewidth=2, alpha=0.8,
                    label=f"Trend {direction} {z[0]:+.2f} %/Tag")
            ax.legend(fontsize=9, framealpha=0.8)

    ax.axvline(1.5, color=C["muted"], linewidth=1,
               linestyle=":", alpha=0.6)
    ax.text(1.55, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
            "Einlaufphase", fontsize=7, color=C["muted"], va="top")
    ax.set_xlabel("Liegetag", fontsize=10, color=C["muted"])
    ax.set_ylabel("Kompressionsrate (%)", fontsize=10, color=C["muted"])
    ax.set_title("Streuung der Kompressionsrate über alle Sensoren pro Liegetag",
                 fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax.set_xlim(0.3, max_day + 0.7)
    plt.tight_layout()
    return fig


def plot_wear_ranking(wear_df, sensors):
    """Horizontales Balkendiagramm: Sensoren nach Gesamtfehlerrate sortiert."""
    C = COLORS

    ranking = []
    for s in sensors:
        sub    = wear_df[wear_df["sensor_id"] == s["sensor_id"]]
        sub_nw = sub[~sub["is_warmup"]] if len(sub[~sub["is_warmup"]]) > 0 else sub
        ranking.append({
            "label": f"#{s['sensor_id']:02d}  {s['start'].strftime('%d.%m.%y')}",
            "comp":  sub_nw["comp_pct"].mean(),
            "roc":   sub_nw["roc_pct"].mean(),
            "plat":  sub_nw["plat_pct"].mean(),
        })
    ranking.sort(key=lambda x: x["comp"] + x["roc"], reverse=True)

    labels = [r["label"] for r in ranking]
    comp_v = np.array([r["comp"] for r in ranking])
    roc_v  = np.array([r["roc"]  for r in ranking])
    plat_v = np.array([r["plat"] for r in ranking])

    # Dynamische Höhe: 0.45cm pro Sensor, min. 5
    rank_h = max(5.0, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(17, rank_h), facecolor=C["bg"])
    ax.set_facecolor("#FAFBFD")
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    ax.tick_params(colors=C["muted"], labelsize=9)

    y = np.arange(len(labels))
    h = max(0.45, min(0.75, 12 / max(len(labels), 1)))

    ax.barh(y, comp_v, h, label="Kompression", color="#EC4899", alpha=0.85)
    ax.barh(y, roc_v,  h, left=comp_v, label="RoC", color=C["danger"], alpha=0.85)
    ax.barh(y, plat_v, h, left=comp_v + roc_v, label="Plateau", color=C["warning"], alpha=0.7)

    ax.set_yticks(y)
    lbl_fs = max(6.0, min(9.0, 280 / max(len(labels), 1)))
    ax.set_yticklabels(labels, fontsize=lbl_fs,
                       color=C["text"], fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Ø Fehlerrate (%)", fontsize=10, color=C["muted"])
    ax.set_title("Sensor-Ranking (schlechteste oben)",
                 fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax.legend(fontsize=9, framealpha=0.8, loc="lower right")

    total_median = float(np.median(comp_v + roc_v))
    ax.axvline(total_median, color=C["muted"], linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.text(total_median + 0.2, len(labels) - 0.5,
            f"Median\n{total_median:.1f}%",
            fontsize=7.5, color=C["muted"], va="bottom")

    for yi, r in zip(y, ranking):
        total = r["comp"] + r["roc"]
        ax.text(total + 0.2, yi, f"{total:.1f}%",
                va="center", fontsize=7.5, color=C["muted"])

    plt.tight_layout()
    return fig


# Alias für Rückwärtskompatibilität (PDF-Report nutzt ggf. den alten Namen)
def plot_wear_time_analysis(wear_df, sensors):
    return plot_wear_heatmap(wear_df, sensors)
    C = COLORS
    n_sensors  = len(sensors)
    max_day    = int(wear_df["day"].max())

    # Dynamische Figurhöhe: min. 0.38cm pro Sensor für die Heatmap
    heat_h   = max(5.0, n_sensors * 0.38)
    total_h  = heat_h + 5.5          # + Raum für die zwei unteren Panels
    fig = plt.figure(figsize=(17, total_h), facecolor=C["bg"])

    # GridSpec: Heatmap oben (proportional zur Sensorzahl), zwei Panels unten fix
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[heat_h, 5.0],
        hspace=0.38, wspace=0.32,
        left=0.13, right=0.97, top=0.95, bottom=0.05,
    )
    ax_heat = fig.add_subplot(gs[0, :])
    ax_box  = fig.add_subplot(gs[1, 0])
    ax_rank = fig.add_subplot(gs[1, 1])

    for ax in [ax_heat, ax_box, ax_rank]:
        ax.set_facecolor("#FAFBFD")
        for spine in ax.spines.values():
            spine.set_color(C["border"])
        ax.tick_params(colors=C["muted"], labelsize=8)

    # ═══════════════════════════════════════════════════════════
    # Panel 1 — HEATMAP  Sensor × Liegetag
    # ═══════════════════════════════════════════════════════════
    # Matrix aufbauen: Zeilen = Sensoren, Spalten = Liegetage 1…max_day
    matrix = np.full((n_sensors, max_day), np.nan)
    ylabels = []
    for row_i, s in enumerate(sensors):
        sub = wear_df[wear_df["sensor_id"] == s["sensor_id"]]
        for _, r in sub.iterrows():
            col_j = int(r["day"]) - 1
            if 0 <= col_j < max_day:
                matrix[row_i, col_j] = r["comp_pct"]
        # Kurzes Label: #Nr  DD.MM.YY
        ylabels.append(f"#{s['sensor_id']:02d}  {s['start'].strftime('%d.%m.%y')}")

    # Farbskala: 0 = hellgrün, 10 = orange, 20+ = rot
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cgm_heat",
        [(0.0, "#E8F5E9"),   # sehr gut
         (0.2, "#FFF9C4"),   # ok
         (0.5, "#FFB300"),   # auffällig
         (1.0, "#C62828")],  # kritisch
    )
    vmax = max(20.0, float(np.nanpercentile(matrix, 95)))

    im = ax_heat.imshow(matrix, aspect="auto", cmap=cmap,
                        vmin=0, vmax=vmax,
                        interpolation="nearest")

    # Farbbalken
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Kompression (%)", fontsize=8, color=C["muted"])
    cbar.ax.tick_params(labelsize=7, colors=C["muted"])

    # Gitter zwischen Sensoren
    ax_heat.set_yticks(np.arange(n_sensors))
    # Dynamische Schriftgröße: bei vielen Sensoren kleiner
    lbl_fs = max(5.0, min(8.5, 220 / max(n_sensors, 1)))
    ax_heat.set_yticklabels(ylabels, fontsize=lbl_fs,
                            color=C["text"], fontfamily="monospace")
    # Abwechselnde Zeilenhinterlegung für bessere Lesbarkeit
    for row_i in range(n_sensors):
        if row_i % 2 == 0:
            ax_heat.axhspan(row_i - 0.5, row_i + 0.5,
                            color="#F0F4F8", alpha=0.35, zorder=0)
    ax_heat.set_xlabel("Liegetag", fontsize=8, color=C["muted"])
    ax_heat.set_title("Kompressionsrate nach Sensor und Liegetag (%)",
                      fontsize=11, fontweight="bold", color=C["text"], pad=10)

    # X-Achse: Tage 1…max_day
    step = max(1, max_day // 15)
    xticks = np.arange(0, max_day, step)
    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels([str(t + 1) for t in xticks], fontsize=7)

    # Einlaufphase (Tag 1) als gestrichelte Linie markieren
    ax_heat.axvline(0.5, color="white", linewidth=1.5,
                    linestyle="--", alpha=0.7, label="Einlaufphase")

    # Horizontale Trennlinien zwischen Sensoren
    for y in np.arange(0.5, n_sensors - 0.5):
        ax_heat.axhline(y, color=C["border"], linewidth=0.4, alpha=0.6)

    # Werte in Zellen: nur wenn Matrix nicht zu groß
    if n_sensors <= 25 and max_day <= 16:
        for row_i in range(n_sensors):
            for col_j in range(max_day):
                v = matrix[row_i, col_j]
                if not np.isnan(v):
                    txt_col = "white" if v > vmax * 0.55 else C["text"]
                    ax_heat.text(col_j, row_i, f"{v:.0f}",
                                 ha="center", va="center",
                                 fontsize=5.5, color=txt_col)

    # ═══════════════════════════════════════════════════════════
    # Panel 2 — BOXPLOT  Streuung der Kompressionsrate pro Liegetag
    # ═══════════════════════════════════════════════════════════
    box_data = []
    box_positions = []
    for day in range(1, max_day + 1):
        vals = wear_df[wear_df["day"] == day]["comp_pct"].dropna().values
        if len(vals) >= 2:
            box_data.append(vals)
            box_positions.append(day)

    if box_data:
        bp = ax_box.boxplot(box_data, positions=box_positions,
                            widths=0.6, patch_artist=True,
                            medianprops=dict(color=C["danger"], linewidth=2),
                            whiskerprops=dict(color=C["muted"], linewidth=1),
                            capprops=dict(color=C["muted"], linewidth=1),
                            flierprops=dict(marker="o", color=C["warning"],
                                            markersize=3, alpha=0.6),
                            boxprops=dict(facecolor=C["primary_lt"],
                                          color=C["primary"], linewidth=1))

        # Median-Linie verbinden
        medians = [np.median(d) for d in box_data]
        ax_box.plot(box_positions, medians, "o-",
                    color=C["primary"], linewidth=1.5,
                    markersize=3, alpha=0.7, zorder=5)

        # Trend-Linie (lineare Regression)
        if len(box_positions) >= 3:
            z = np.polyfit(box_positions, medians, 1)
            p = np.poly1d(z)
            x_trend = np.array([box_positions[0], box_positions[-1]])
            ax_box.plot(x_trend, p(x_trend), "--",
                        color=C["danger"], linewidth=1.5, alpha=0.8,
                        label=f"Trend ({z[0]:+.2f}%/Tag)")
            ax_box.legend(fontsize=7, framealpha=0.8)

    ax_box.axvline(1.5, color=C["muted"], linewidth=1,
                   linestyle=":", alpha=0.6, label="nach Einlaufphase")
    ax_box.set_xlabel("Liegetag", fontsize=8, color=C["muted"])
    ax_box.set_ylabel("Kompressionsrate (%)", fontsize=8, color=C["muted"])
    ax_box.set_title("Streuung über alle Sensoren pro Liegetag",
                     fontsize=10, fontweight="bold", color=C["text"], pad=8)
    ax_box.set_xlim(0.3, max_day + 0.7)

    # ═══════════════════════════════════════════════════════════
    # Panel 3 — RANKING  Sensoren nach Ø Gesamtfehlerrate
    # ═══════════════════════════════════════════════════════════
    ranking = []
    for s in sensors:
        sub = wear_df[wear_df["sensor_id"] == s["sensor_id"]]
        sub_nw = sub[~sub["is_warmup"]] if len(sub[~sub["is_warmup"]]) > 0 else sub
        ranking.append({
            "label":  f"#{s['sensor_id']} {s['start'].strftime('%d.%m.%y')}",
            "comp":   sub_nw["comp_pct"].mean(),
            "roc":    sub_nw["roc_pct"].mean(),
            "plat":   sub_nw["plat_pct"].mean(),
        })

    # Nach Kompression + RoC sortieren (schlechteste oben)
    ranking.sort(key=lambda x: x["comp"] + x["roc"], reverse=True)

    labels  = [r["label"] for r in ranking]
    comp_v  = np.array([r["comp"] for r in ranking])
    roc_v   = np.array([r["roc"]  for r in ranking])
    plat_v  = np.array([r["plat"] for r in ranking])

    y = np.arange(len(labels))
    h = max(0.55, min(0.8, 12 / max(len(labels), 1)))

    ax_rank.barh(y, comp_v, h, label="Kompression", color="#EC4899", alpha=0.85)
    ax_rank.barh(y, roc_v,  h, left=comp_v, label="RoC", color=C["danger"], alpha=0.85)
    ax_rank.barh(y, plat_v, h, left=comp_v + roc_v,
                 label="Plateau", color=C["warning"], alpha=0.7)

    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels, fontsize=7.5, color=C["text"])
    ax_rank.set_xlabel("Ø Fehlerrate (%)", fontsize=8, color=C["muted"])
    ax_rank.set_title("Sensor-Ranking (schlechteste oben)",
                      fontsize=10, fontweight="bold", color=C["text"], pad=8)
    ax_rank.invert_yaxis()   # schlechteste (Ende der sortierten Liste) nach oben
    ax_rank.legend(fontsize=7, framealpha=0.8, loc="lower right")

    # Median-Linie
    total_median = float(np.median(comp_v + roc_v))
    ax_rank.axvline(total_median, color=C["muted"], linewidth=1,
                    linestyle="--", alpha=0.7,
                    label=f"Median {total_median:.1f}%")

    # Werte am Balkenende
    for yi, r in zip(y, ranking):
        total = r["comp"] + r["roc"]
        ax_rank.text(total + 0.2, yi, f"{total:.1f}%",
                     va="center", fontsize=6.5, color=C["muted"])

    plt.suptitle("Sensor-Liegezeit-Analyse", fontsize=13,
                 fontweight="bold", color=C["text"], y=1.01)
    return fig


# ─────────────────────────────────────────────────────────────
#  PDF-REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_pdf_report(df, meta, score, gap_count, wear_df=None, sensors=None):
    """Erstellt einen professionellen PDF-Report im Light-Theme."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage,
                                         HRFlowable, KeepTogether)
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    except ImportError:
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=2*cm, bottomMargin=2*cm,
                             leftMargin=2.2*cm, rightMargin=2.2*cm)

    # ── Farben (Light-Theme)
    CLR_PRIMARY  = colors.HexColor("#0071BC")
    CLR_BG       = colors.HexColor("#FFFFFF")
    CLR_SURFACE  = colors.HexColor("#F7F9FC")
    CLR_TEXT     = colors.HexColor("#1A202C")
    CLR_MUTED    = colors.HexColor("#4A5568")   # dunkleres Grau für bessere Lesbarkeit
    CLR_BORDER   = colors.HexColor("#CBD5E0")
    CLR_FLAG     = colors.HexColor("#DC2626")
    CLR_WARN     = colors.HexColor("#D97706")
    CLR_OK       = colors.HexColor("#16A34A")

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle("title",  fontSize=22, fontName="Helvetica-Bold",
                                  textColor=CLR_PRIMARY, spaceAfter=4, leading=26)
    style_sub   = ParagraphStyle("sub",    fontSize=11, fontName="Helvetica",
                                  textColor=CLR_MUTED, spaceAfter=4)
    style_h2    = ParagraphStyle("h2",     fontSize=13, fontName="Helvetica-Bold",
                                  textColor=CLR_PRIMARY, spaceBefore=14, spaceAfter=6)
    style_body  = ParagraphStyle("body",   fontSize=9,  fontName="Helvetica",
                                  textColor=CLR_TEXT, leading=14, spaceAfter=4)
    style_muted = ParagraphStyle("muted",  fontSize=8,  fontName="Helvetica",
                                  textColor=CLR_MUTED, leading=12)

    # Wiederverwendbarer Tabellenstil (Light)
    def make_tbl_style(col_widths=None):
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  CLR_PRIMARY),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("BACKGROUND",   (0, 1), (-1, -1), CLR_BG),
            ("ROWBACKGROUNDS",(0,1), (-1, -1), [CLR_BG, CLR_SURFACE]),
            ("TEXTCOLOR",    (0, 1), (-1, -1), CLR_TEXT),
            ("GRID",         (0, 0), (-1, -1), 0.3, CLR_BORDER),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ])

    def fig_to_image(fig, w=16*cm, h=5*cm):
        imgbuf = BytesIO()
        fig.savefig(imgbuf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#FFFFFF")
        imgbuf.seek(0)
        return RLImage(imgbuf, width=w, height=h)

    elements = []

    # ── Header
    elements.append(Paragraph("CGM Validator", style_title))
    elements.append(Paragraph("Qualitätsanalyse kontinuierlicher Glukosemessung", style_sub))
    elements.append(Paragraph(
        f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')} &nbsp;|&nbsp; "
        f"Quelle: {meta['source']} &nbsp;|&nbsp; Intervall: {meta['interval']} min",
        style_body))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=CLR_PRIMARY, spaceAfter=10))

    # ── Qualitätsscore
    if score >= 90:
        score_color, score_label = CLR_OK,   "Sehr gut"
    elif score >= 70:
        score_color, score_label = CLR_WARN, "Akzeptabel"
    else:
        score_color, score_label = CLR_FLAG, "Kritisch"

    score_style = ParagraphStyle("score", fontSize=36, fontName="Helvetica-Bold",
                                  textColor=score_color, spaceAfter=2, leading=40)
    elements.append(Paragraph("Gesamt-Qualitätsscore", style_h2))
    elements.append(Paragraph(f"{score:.1f} / 100 — {score_label}", score_style))
    elements.append(Spacer(1, 0.3*cm))

    # ── Datensatz-Übersicht
    elements.append(Paragraph("Datensatz-Übersicht", style_h2))
    overview_data = [
        ["Parameter", "Wert"],
        ["Zeitraum",            f"{meta['start']} – {meta['end']}"],
        ["Gesamtmessungen",     f"{meta['n']:,}"],
        ["Messintervall",       f"{meta['interval']} Minute(n)"],
        ["Erwartete Messungen", f"{meta['expected_n']:,}"],
        ["Verfügbarkeit",       f"{meta['availability']:.1f}%"],
    ]
    tbl = Table(overview_data, colWidths=[7*cm, 9*cm])
    tbl.setStyle(make_tbl_style())
    elements.append(tbl)
    elements.append(Spacer(1, 0.5*cm))

    # ── Analyse-Ergebnisse
    elements.append(Paragraph("Analyse-Ergebnisse", style_h2))
    details_data = [
        ["Modul", "Befunde", "Anteil", "Bewertung"],
        ["Rate-of-Change",
         str(df["roc_flag"].sum()),
         f"{100*df['roc_flag'].sum()/max(meta['n'],1):.2f}%",
         "⚠ Kritisch" if df["roc_flag"].sum() > meta["n"]*0.01 else "✓ OK"],
        ["Zeitlücken",
         str(gap_count), "—",
         "⚠ Prüfen" if gap_count > 5 else "✓ OK"],
        ["Plateau-Detektor",
         str(df["plateau_flag"].sum()),
         f"{100*df['plateau_flag'].sum()/max(meta['n'],1):.2f}%",
         "⚠ Prüfen" if df["plateau_flag"].sum() > 0 else "✓ OK"],
        ["Stat. Ausreißer (z>3.5)",
         str(df["zscore_flag"].sum()),
         f"{100*df['zscore_flag'].sum()/max(meta['n'],1):.2f}%",
         "⚠ Prüfen" if df["zscore_flag"].sum() > meta["n"]*0.005 else "✓ OK"],
        ["Kompressionsartefakte",
         str(df["compression_flag"].sum()),
         f"{100*df['compression_flag'].sum()/max(meta['n'],1):.2f}%",
         "⚠ Prüfen" if df["compression_flag"].sum() > 0 else "✓ OK"],
    ]
    dtbl = Table(details_data, colWidths=[5*cm, 3*cm, 3*cm, 5*cm])
    dtbl.setStyle(make_tbl_style())
    elements.append(dtbl)
    elements.append(Spacer(1, 0.5*cm))

    # ── Glukose-Zeitverlauf
    elements.append(Paragraph("Glukose-Zeitverlauf", style_h2))
    fig_tl = plot_glucose_timeline(df)
    elements.append(fig_to_image(fig_tl, w=16*cm, h=4.5*cm))
    plt.close(fig_tl)

    # ── Poincaré + Deltaverteilung
    elements.append(Paragraph("Poincaré-Plot & Änderungsraten-Verteilung", style_h2))
    fig_pc = plot_poincare(df)
    fig_dd = plot_delta_distribution(df, meta["threshold"])
    elements.append(fig_to_image(fig_pc, w=7.5*cm, h=7*cm))
    elements.append(fig_to_image(fig_dd, w=7.5*cm, h=5*cm))
    plt.close(fig_pc)
    plt.close(fig_dd)

    # ── Sensor-Liegezeit-Analyse (wenn vorhanden)
    if wear_df is not None and sensors is not None and len(sensors) > 0:
        elements.append(Paragraph("Sensor-Liegezeit-Analyse", style_h2))

        # Heatmap
        elements.append(Paragraph("Kompressionsrate nach Sensor und Liegetag", style_muted))
        fig_heat = plot_wear_heatmap(wear_df, sensors)
        heat_h = max(3*cm, min(12*cm, len(sensors) * 0.38 * cm))
        elements.append(fig_to_image(fig_heat, w=16*cm, h=heat_h))
        plt.close(fig_heat)
        elements.append(Spacer(1, 0.3*cm))

        # Boxplot
        elements.append(Paragraph("Streuung der Kompressionsrate pro Liegetag", style_muted))
        fig_box = plot_wear_boxplot(wear_df)
        elements.append(fig_to_image(fig_box, w=16*cm, h=5*cm))
        plt.close(fig_box)
        elements.append(Spacer(1, 0.3*cm))

        # Ranking
        elements.append(Paragraph("Sensor-Ranking nach Gesamtfehlerrate", style_muted))
        fig_rank = plot_wear_ranking(wear_df, sensors)
        rank_h = max(4*cm, min(14*cm, len(sensors) * 0.45 * cm))
        elements.append(fig_to_image(fig_rank, w=16*cm, h=rank_h))
        plt.close(fig_rank)

    # ── Footer
    elements.append(Spacer(1, 1*cm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=CLR_BORDER))
    elements.append(Paragraph(
        "CGM Validator v1.1 — Lokal erstellt. Keine Daten verlassen diesen Rechner. "
        "Dieses Dokument dient der technischen Qualitätsbewertung und ersetzt keine klinische Beurteilung.",
        ParagraphStyle("footer", fontSize=7, fontName="Helvetica",
                        textColor=CLR_MUTED, leading=10)))

    doc.build(elements)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
#  STREAMLIT APP — HAUPTLAYOUT
# ─────────────────────────────────────────────────────────────

# ── Sidebar
with st.sidebar:
    st.markdown("## 🩺 CGM Validator")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Version 1.1**")
    st.markdown('<div class="info-card">Alle Daten bleiben lokal auf diesem Rechner. Keine Cloud-Übertragung.</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("#### Analyse-Einstellungen")
    custom_threshold = st.checkbox("RoC-Schwellenwert überschreiben", value=False, key="cb_roc")
    manual_threshold = None
    if custom_threshold:
        manual_threshold = st.slider("Max. Δ Glukose (mg/dL)", 1, 40, 4, key="sl_roc")

    custom_plateau = st.checkbox("Plateau-Schwelle überschreiben", value=False, key="cb_plateau")
    manual_plateau = None
    if custom_plateau:
        manual_plateau = st.slider(
            "Min. identische Werte = Plateau",
            min_value=2, max_value=30, value=6,
            help=(
                "Standard-Algorithmus (ohne Überschreiben):\n"
                "• 1-min-Daten: ≥20 Werte (20 min) immer | ≥8 Werte + Δ>4 mg/dL kontextuell\n"
                "• 5-min-Daten: ≥6 Werte (30 min) + Δ>8 mg/dL\n\n"
                "Beim manuellen Überschreiben: nur Länge zählt, Kontext-Sprung entfällt."
            ),
            key="sl_plateau"
        )

    show_raw = st.checkbox("Rohdaten anzeigen", value=False, key="cb_raw")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Unterstützte Quellen")
    for src in ["Glooko ZIP (bis zu 6 CGM-Dateien)", "Glooko / FreeStyle Libre 2/3",
                "Dexcom Clarity", "Abbott LibreView",
                "Medtronic CareLink", "Nightscout", "Unbekannte CSV (Heuristik)"]:
        st.markdown(f"<small>✓ {src}</small>", unsafe_allow_html=True)


# ── Hauptbereich
st.markdown("# CGM Validator")
st.markdown("**Qualitätsanalyse kontinuierlicher Glukosemessung**")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "CGM-Dateien hochladen — CSV, TXT oder ZIP (Glooko-Export)",
    type=["csv", "txt", "zip"],
    accept_multiple_files=True,
    help="Eine oder mehrere Dateien: einzelne CSV/TXT oder Glooko-ZIPs. "
         "Mehrere ZIPs vom selben Patienten werden automatisch zu einer Zeitreihe zusammengeführt."
)

if not uploaded_files:
    st.markdown("""
    <div class="info-card">
    <strong>Wie es funktioniert:</strong><br>
    1. CSV/TXT aus Ihrem CGM-Portal exportieren — oder Glooko-ZIP(s) direkt hochladen<br>
    2. <strong>Mehrere ZIPs gleichzeitig wählbar</strong> — alle CGM-Dateien werden automatisch
       chronologisch zusammengeführt und dedupliziert (ideal für lange Beobachtungszeiträume)<br>
    3. Analyse startet sofort — Ergebnisse erscheinen in den Tabs unten<br>
    4. PDF-Report exportieren
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Analyse-Module")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card"><strong>📈 Rate-of-Change</strong><br>Erkennt physiologisch unmögliche Sprünge (&gt;4 mg/dL/min)</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><strong>📊 Poincaré-Plot</strong><br>Visualisiert konsekutive Wert-Paare — Artefakte fallen sofort auf</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><strong>⏱ Zeitlücken</strong><br>Erkennt Sensorausfälle und fehlende Messungen</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><strong>📉 Plateaus</strong><br>Signal-Sticking: identische Werte über mehrere Minuten</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><strong>🔬 Stat. Ausreißer</strong><br>Lokaler Z-Score im 30-min-Fenster</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><strong>😴 Kompression</strong><br>Schneller Abfall + Erholung (Schlafen auf Sensor)</div>', unsafe_allow_html=True)
    st.stop()


# ── Datei laden & analysieren ────────────────────────────────────
import zipfile as _zipfile

def _parse_zip(uploaded_zip):
    """
    Extrahiert alle CGM-CSVs aus einem ZIP, parst und gibt
    (segments[], source_name, n_csv_found) zurück.
    """
    raw_bytes = uploaded_zip.read()
    try:
        zf = _zipfile.ZipFile(BytesIO(raw_bytes))
    except Exception as e:
        return [], "Unbekannt", 0

    # CGM-Kandidaten: Glooko-Muster CGM_data_*.csv, Unterordner möglich
    cgm_names = sorted([
        n for n in zf.namelist()
        if re.search(r'cgm[_\s-]?data[_\s-]?\d*\.(csv|txt)$', n, re.IGNORECASE)
        or re.search(r'cgm.*\.(csv|txt)$', n, re.IGNORECASE)
    ])
    if not cgm_names:
        cgm_names = sorted([n for n in zf.namelist()
                             if n.lower().endswith(".csv") and "__MACOSX" not in n])

    segments, source_name = [], "Unbekannt"
    for zname in cgm_names:
        try:
            bio = BytesIO(zf.read(zname))
            bio.name = zname.split("/")[-1]
            df_seg, _, _, src, err = parse_cgm_csv(bio)
            if err or df_seg is None or len(df_seg) < 5:
                continue
            segments.append(df_seg)
            if source_name == "Unbekannt":
                source_name = src
        except Exception:
            continue

    return segments, source_name, len(cgm_names)


def load_all_uploads(uploaded_files):
    """
    Verarbeitet eine Liste von Uploads (CSV/TXT/ZIP, beliebig gemischt).
    Alle Segmente werden chronologisch sortiert, zusammengeführt und
    dedupliziert → eine einzige Zeitreihe.
    Gibt (df, source_name, error, info_dict) zurück.
    """
    all_segments = []
    source_name  = "Unbekannt"
    n_zips = n_csvs = n_cgm_files = 0

    for uf in uploaded_files:
        fname = uf.name.lower()

        if fname.endswith(".zip"):
            segs, src, n_found = _parse_zip(uf)
            all_segments.extend(segs)
            n_zips += 1
            n_cgm_files += n_found
            if source_name == "Unbekannt" and src != "Unbekannt":
                source_name = src

        elif fname.endswith(".csv") or fname.endswith(".txt"):
            try:
                uf.seek(0)
            except Exception:
                pass
            df_seg, _, _, src, err = parse_cgm_csv(uf)
            if not err and df_seg is not None and len(df_seg) >= 5:
                all_segments.append(df_seg)
                n_csvs += 1
                n_cgm_files += 1
                if source_name == "Unbekannt":
                    source_name = src
            else:
                pass  # Segment wird stillschweigend übersprungen

    if not all_segments:
        return None, source_name, "Keine verwertbaren CGM-Daten gefunden.", {}

    # Chronologisch sortieren nach erstem Zeitstempel je Segment
    all_segments.sort(key=lambda d: d["datetime"].iloc[0])

    # Zusammenführen + Duplikate entfernen (Glooko-Exporte überlappen sich)
    df_all = pd.concat(all_segments, ignore_index=True)
    before_dedup = len(df_all)
    df_all = (df_all
              .sort_values("datetime")
              .drop_duplicates(subset=["datetime"])
              .reset_index(drop=True))
    n_dupes = before_dedup - len(df_all)

    # Flag-Spalten sicherstellen
    for col in ["roc_flag", "plateau_flag", "zscore_flag",
                "compression_flag", "gap_flag", "delta"]:
        if col not in df_all.columns:
            df_all[col] = False if "flag" in col else 0.0

    info = {
        "n_zips":      n_zips,
        "n_csvs":      n_csvs,
        "n_cgm_files": n_cgm_files,
        "n_segments":  len(all_segments),
        "n_dupes":     n_dupes,
        "span_days":   (df_all["datetime"].iloc[-1] -
                        df_all["datetime"].iloc[0]).total_seconds() / 86400,
    }
    return df_all, source_name, None, info


with st.spinner("Dateien werden eingelesen und zusammengeführt…"):
    df, source_name, error, load_info = load_all_uploads(uploaded_files)

# ── Lade-Info-Banner
if load_info:
    parts = []
    if load_info["n_zips"] > 0:
        parts.append(f"<strong>{load_info['n_zips']} ZIP(s)</strong> mit "
                     f"{load_info['n_cgm_files']} CGM-Dateien")
    if load_info["n_csvs"] > 0:
        parts.append(f"<strong>{load_info['n_csvs']} CSV(s)</strong>")
    span = load_info.get("span_days", 0)
    dupes = load_info.get("n_dupes", 0)
    dupe_note = f" · {dupes:,} Duplikate entfernt" if dupes > 0 else ""
    st.markdown(
        f'<div class="info-card">📦 {" + ".join(parts)} → '
        f'<strong>{load_info["n_segments"]} Segmente</strong> zusammengeführt · '
        f'<strong>{span:.0f} Tage</strong> Gesamtzeitraum{dupe_note}</div>',
        unsafe_allow_html=True
    )

if error:
    st.markdown(f'<div class="danger-card">❌ <strong>Fehler beim Einlesen:</strong> {error}</div>',
                unsafe_allow_html=True)
    st.stop()

if df is None or len(df) < 10:
    st.markdown('<div class="danger-card">❌ Zu wenige Datenpunkte für eine Analyse (min. 10 benötigt).</div>',
                unsafe_allow_html=True)
    st.stop()

# Intervall erkennen
interval_min = detect_interval(df)

# Schwellenwert
if manual_threshold:
    threshold_val = manual_threshold
else:
    threshold_val = 4.0 if interval_min == 1 else 20.0

# Alle Analyse-Module
df, threshold = analyze_rate_of_change(df, interval_min)
if manual_threshold:
    threshold = manual_threshold
    df["roc_flag"] = df["delta"] > threshold

df, gaps_df = analyze_gaps(df, interval_min)
df = analyze_plateaus(df, interval_min, manual_plateau)
df = analyze_statistical_outliers(df, window_min=30, interval_min=interval_min)
df = analyze_compression(df, interval_min)

# ── Sensor-Erkennung & Liegezeit-Analyse
_max_wear, _warmup_h, _gap_thresh = get_sensor_config(source_name)
sensors = detect_sensors(df, gap_thresh_min=_gap_thresh, max_wear_days=_max_wear)
wear_df = analyze_wear_time(df, sensors, warmup_h=_warmup_h) if sensors else pd.DataFrame()


total_duration = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 60
expected_n = int(total_duration / interval_min) + 1
availability = 100 * len(df) / max(expected_n, 1)

meta = {
    "source":      source_name,
    "interval":    interval_min,
    "n":           len(df),
    "start":       df["datetime"].iloc[0].strftime("%d.%m.%Y %H:%M"),
    "end":         df["datetime"].iloc[-1].strftime("%d.%m.%Y %H:%M"),
    "expected_n":  expected_n,
    "availability": availability,
    "threshold":   threshold,
}

score = compute_quality_score(df, gaps_df, interval_min)

# ── Quell-Badge
source_color = {
    "Glooko / FreeStyle Libre": COLORS["primary"],
    "Dexcom Clarity":           "#0E7490",
    "Abbott LibreView":         "#0369A1",
    "Medtronic CareLink":       "#7C3AED",
    "Nightscout":               "#059669",
}.get(source_name, COLORS["warning"])

st.markdown(
    f'<span style="background:{source_color};color:#000;padding:3px 10px;'
    f'border-radius:4px;font-size:0.8rem;font-weight:700;">'
    f'{source_name}</span>&nbsp;&nbsp;'
    f'<span style="color:{COLORS["muted"]};font-size:0.85rem;">'
    f'{"ZIP · " + str(load_info.get("n_cgm_files",0)) + " Dateien zusammengeführt" if load_info.get("n_zips",0) > 0 else "Erkannte Quelle: " + source_name}</span>',
    unsafe_allow_html=True
)
st.markdown("")

# ── KPI-Metriken
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Messungen", f"{len(df):,}")
c2.metric("Zeitraum", f"{meta['start'][:10]} → {meta['end'][:10]}")
c3.metric("Intervall", f"{interval_min} min")
c4.metric("Verfügbarkeit", f"{availability:.1f}%")

score_delta = "🟢 Sehr gut" if score >= 90 else ("🟡 OK" if score >= 70 else "🔴 Kritisch")
c5.metric("Qualitätsscore", f"{score:.1f}/100", score_delta)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Zeitverlauf", "🔬 Poincaré", "📊 Statistik",
    "🚨 Fehler-Detail", "🩹 Liegezeit", "📄 Report"
])

with tab1:
    # ── Zeitraum-Auswahl ──────────────────────────────────────
    n_days = df["datetime"].dt.date.nunique()
    all_dates = sorted(df["datetime"].dt.date.unique())
    date_min = all_dates[0]
    date_max = all_dates[-1]

    if n_days > 1:
        selected_range = st.slider(
            "Angezeigter Zeitraum",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_max),
            format="DD.MM.YY",
            key="sl_daterange",
        )
        # Wenn Start == End: genau diesen einen Tag anzeigen (00:00–23:59)
        start_dt = pd.Timestamp(selected_range[0])
        if selected_range[0] == selected_range[1]:
            end_dt = start_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            end_dt = pd.Timestamp(selected_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        df_view = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
        if len(df_view) == 0:
            st.warning("Keine Daten im gewählten Zeitraum.")
            df_view = df
    else:
        df_view = df

    # ── Zeitverlauf-Plot ──────────────────────────────────────
    fig_tl = plot_glucose_timeline(df_view)
    st.pyplot(fig_tl, use_container_width=True)
    plt.close(fig_tl)

    # Tages-Qualitätsübersicht (immer über Gesamtdaten)
    if n_days >= 2:
        st.markdown("")
        fig_dq = plot_daily_quality(df)
        st.pyplot(fig_dq, use_container_width=True)
        plt.close(fig_dq)

    # Fehler-Zusammenfassung (immer über Gesamtdaten)
    st.markdown("")
    n_roc  = int(df["roc_flag"].sum())
    n_plat = int(df["plateau_flag"].sum())
    n_stat = int(df["zscore_flag"].sum())
    n_comp = int(df["compression_flag"].sum())
    n_gap  = len(gaps_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔴 RoC-Fehler",       n_roc,  f"{100*n_roc/len(df):.2f}%")
    c2.metric("🟡 Plateaus",         n_plat, f"{100*n_plat/len(df):.2f}%")
    c3.metric("🟣 Stat. Ausreißer",  n_stat, f"{100*n_stat/len(df):.2f}%")
    c4.metric("🩷 Kompression",      n_comp, f"{100*n_comp/len(df):.2f}%")
    c5.metric("⬜ Zeitlücken",       n_gap)

    # Medtronic-Hinweis
    if "Closed-Loop" in source_name:
        st.markdown("""
        <div class="warn-card">
        ⚠ <strong>Medtronic Closed-Loop Export:</strong>
        Die CGM-Werte wurden aus der <code>Bolus Number</code>-Spalte extrahiert
        (Closed-Loop CGM-Referenzwert je Mikro-Bolus). Zeitstempel entsprechen den
        Bolus-Ereignissen, nicht exakten 5-min-CGM-Intervallen — minimale
        Zeitunregelmäßigkeiten sind daher normal.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pc = plot_poincare(df)
        st.pyplot(fig_pc, use_container_width=True)
        plt.close(fig_pc)
    with col_b:
        fig_dd = plot_delta_distribution(df, threshold)
        st.pyplot(fig_dd, use_container_width=True)
        plt.close(fig_dd)

    st.markdown(f"""
    <div class="info-card">
    <strong>Poincaré-Plot erklärt:</strong> Jeder Punkt zeigt G(t) gegen G(t+1).
    Punkte auf der Diagonalen = keine Änderung zwischen zwei Messungen.
    Die grüne Ellipse zeigt <strong>SD1</strong> (kurzfristige Variabilität, quer zur Diagonalen)
    und <strong>SD2</strong> (Langzeit-Trend, entlang der Diagonalen).
    Rote Punkte = Sprünge über {threshold:.0f} mg/dL — physiologisch implausibel.
    <br><br>
    <strong>Änderungsraten-Histogramm:</strong> Idealerweise fallen fast alle Werte links
    der roten Schwelle. Ein langer Schwanz rechts der Schwelle deutet auf systematische
    Artefakte hin.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    fig_z = plot_zscore_timeline(df)
    st.pyplot(fig_z, use_container_width=True)
    plt.close(fig_z)

    # Deskriptive Statistik
    st.markdown("#### Deskriptive Statistik & Time-in-Range")
    glucose = df["glucose"]
    col1, col2 = st.columns(2)
    with col1:
        # GMI (Glucose Management Indicator) ≈ HbA1c-Schätzung
        gmi = 3.31 + 0.02392 * glucose.mean()
        cv  = 100 * glucose.std() / glucose.mean()
        stats_data = pd.DataFrame({
            "Kennzahl": ["Mittelwert", "Median", "Std.-Abweichung", "VK (%)",
                          "Min", "Max", "10. Perz.", "90. Perz.", "IQR",
                          "GMI (≈ HbA1c %)", "GMI (≈ mmol/mol)"],
            "Wert": [
                f"{glucose.mean():.1f} mg/dL",
                f"{glucose.median():.1f} mg/dL",
                f"{glucose.std():.1f} mg/dL",
                f"{cv:.1f}%  {'⚠ > 36%' if cv > 36 else '✓'}",
                f"{glucose.min():.0f} mg/dL",
                f"{glucose.max():.0f} mg/dL",
                f"{glucose.quantile(0.10):.1f} mg/dL",
                f"{glucose.quantile(0.90):.1f} mg/dL",
                f"{glucose.quantile(0.75) - glucose.quantile(0.25):.1f} mg/dL",
                f"{gmi:.1f}%",
                f"{(gmi - 2.15) / 0.0915:.0f} mmol/mol",
            ]
        })
        st.dataframe(stats_data, hide_index=True, use_container_width=True)
        st.markdown(
            '<div class="info-card" style="font-size:0.78rem;">'
            '<strong>GMI</strong> = Glucose Management Indicator (Schätzung des HbA1c aus CGM-Daten). '
            '<strong>VK</strong> = Variationskoeffizient (Zielwert &lt; 36%).</div>',
            unsafe_allow_html=True
        )
    with col2:
        tir  = 100 * ((glucose >= 70) & (glucose <= 180)).mean()
        tbr1 = 100 * ((glucose >= 54) & (glucose < 70)).mean()
        tbr2 = 100 * (glucose < 54).mean()
        tar1 = 100 * ((glucose > 180) & (glucose <= 250)).mean()
        tar2 = 100 * (glucose > 250).mean()
        tir_df = pd.DataFrame({
            "Zone": [
                "🟢 TIR (70–180 mg/dL)",
                "🟡 TBR L1 (54–70 mg/dL)",
                "🔴 TBR L2 (< 54 mg/dL)",
                "🟡 TAR L1 (180–250 mg/dL)",
                "🔴 TAR L2 (> 250 mg/dL)"
            ],
            "Anteil": [f"{tir:.1f}%", f"{tbr1:.1f}%", f"{tbr2:.1f}%",
                       f"{tar1:.1f}%", f"{tar2:.1f}%"],
            "Ziel (AGC)": ["≥ 70%", "< 4%", "< 1%", "< 25%", "< 5%"],
            "Status": [
                "✓" if tir >= 70 else "⚠",
                "✓" if tbr1 < 4 else "⚠",
                "✓" if tbr2 < 1 else "⚠",
                "✓" if tar1 < 25 else "⚠",
                "✓" if tar2 < 5 else "⚠",
            ]
        })
        st.dataframe(tir_df, hide_index=True, use_container_width=True)
        st.markdown(
            '<div class="info-card" style="font-size:0.78rem;">'
            'Zielwerte nach <strong>AGC/ATTD Consensus 2019</strong> für Typ-1-Diabetes. '
            'Für andere Populationen können abweichende Ziele gelten.</div>',
            unsafe_allow_html=True
        )

    # Zeitlücken-Detail
    if len(gaps_df) > 0:
        st.markdown("#### Erkannte Zeitlücken")
        gaps_display = gaps_df[["datetime", "gap_minutes"]].copy()
        gaps_display.columns = ["Zeitpunkt nach Lücke", "Lücke (Minuten)"]
        gaps_display["Lücke (Minuten)"] = gaps_display["Lücke (Minuten)"].round(1)
        # Lücken nach Länge kategorisieren
        gaps_display["Kategorie"] = gaps_display["Lücke (Minuten)"].apply(
            lambda m: "Kurz (< 30 min)" if m < 30 else ("Mittel (30–120 min)" if m < 120 else "Lang (> 2h)")
        )
        st.dataframe(gaps_display, hide_index=True, use_container_width=True)

with tab4:
    st.markdown("#### Alle flagged Messungen")
    flag_mask = (df["roc_flag"] | df["plateau_flag"] |
                 df["zscore_flag"] | df["compression_flag"])
    flagged_df = df[flag_mask].copy()

    if len(flagged_df) == 0:
        st.markdown('<div class="info-card">✅ Keine Fehlmessungen in diesem Datensatz gefunden.</div>',
                    unsafe_allow_html=True)
    else:
        # Zusammenfassung oben
        n_only_roc   = int((df["roc_flag"] & ~df["plateau_flag"] & ~df["zscore_flag"]).sum())
        n_only_plat  = int((df["plateau_flag"] & ~df["roc_flag"]).sum())
        n_combined   = int((df["roc_flag"] & df["plateau_flag"]).sum())

        st.markdown(f"""
        <div class="warn-card">
        <strong>{len(flagged_df):,} flagged Messungen</strong> von {len(df):,} gesamt
        ({100*len(flagged_df)/len(df):.1f}%)
        — RoC-Fehler: <strong>{int(df['roc_flag'].sum())}</strong>
        &nbsp;|&nbsp; Plateaus: <strong>{int(df['plateau_flag'].sum())}</strong>
        &nbsp;|&nbsp; Stat. Ausreißer: <strong>{int(df['zscore_flag'].sum())}</strong>
        &nbsp;|&nbsp; Kompression: <strong>{int(df['compression_flag'].sum())}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Filter-Optionen
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            filter_type = st.multiselect(
                "Filter nach Fehlertyp:",
                ["RoC-Fehler", "Plateau", "Stat. Ausreißer", "Kompression"],
                default=["RoC-Fehler", "Plateau", "Stat. Ausreißer", "Kompression"]
            )
        with col_f2:
            max_rows = st.selectbox("Max. Zeilen", [100, 500, 1000, "Alle"], index=1)

        # Maske nach Filter
        filter_mask = pd.Series(False, index=df.index)
        if "RoC-Fehler"      in filter_type: filter_mask |= df["roc_flag"]
        if "Plateau"         in filter_type: filter_mask |= df["plateau_flag"]
        if "Stat. Ausreißer" in filter_type: filter_mask |= df["zscore_flag"]
        if "Kompression"     in filter_type: filter_mask |= df["compression_flag"]

        filtered = df[filter_mask].copy()
        if max_rows != "Alle":
            filtered = filtered.head(int(max_rows))

        display = pd.DataFrame({
            "Zeitstempel":     filtered["datetime"].dt.strftime("%d.%m.%Y %H:%M"),
            "Glukose (mg/dL)": filtered["glucose"].round(1),
            "Δ (mg/dL)":      filtered["delta"].round(1),
            "RoC":             filtered["roc_flag"].map({True: "⚠", False: ""}),
            "Plateau":         filtered["plateau_flag"].map({True: "⚠", False: ""}),
            "Z-Score":         filtered["zscore_flag"].map({True: "⚠", False: ""}),
            "Kompression":     filtered["compression_flag"].map({True: "⚠", False: ""}),
        })
        st.dataframe(display, hide_index=True, use_container_width=True)

        csv_export = df[flag_mask][["datetime","glucose","delta","roc_flag",
                                    "plateau_flag","zscore_flag","compression_flag"]]
        csv_bytes = csv_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Alle flagged Messungen als CSV exportieren",
            data=csv_bytes,
            file_name=f"cgm_flagged_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    if show_raw:
        st.markdown("#### Rohdaten (erste 500 Zeilen)")
        st.dataframe(df[["datetime", "glucose", "delta"]].head(500), use_container_width=True)

with tab5:
    st.markdown("### 🩹 Sensor-Liegezeit-Analyse")

    if not sensors:
        st.info("Keine Sensor-Wechsel erkannt — zu wenig Daten oder nur ein Sensor im Zeitraum.")
    elif wear_df.empty:
        st.info("Nicht genügend Daten für die Liegezeit-Analyse.")
    else:
        # ── Sensor-Übersicht Tabelle
        st.markdown("**Erkannte Sensoren**")
        sensor_rows = []
        for s in sensors:
            sub = wear_df[wear_df["sensor_id"] == s["sensor_id"]]
            sub_nw = sub[~sub["is_warmup"]] if "is_warmup" in sub.columns else sub
            if len(sub_nw) == 0:
                sub_nw = sub
            sensor_rows.append({
                "Sensor": f"#{s['sensor_id']}",
                "Start":  s["start"].strftime("%d.%m.%Y %H:%M"),
                "Ende":   s["end"].strftime("%d.%m.%Y %H:%M"),
                "Liegedauer": f"{s['wear_days']:.1f} Tage ({s['wear_hours']:.0f}h)",
                "Messungen": f"{s['n']:,}",
                "Ø RoC-Fehler": f"{sub_nw['roc_pct'].mean():.1f}%",
                "Ø Kompression": f"{sub_nw['comp_pct'].mean():.1f}%",
                "Nacht-Kompr.": f"{sub_nw['night_comp_pct'].mean():.1f}%" if "night_comp_pct" in sub_nw.columns and sub_nw['night_comp_pct'].notna().any() else "—",
                "Erkannt via": s.get("split_reason", "—"),
            })
        st.dataframe(pd.DataFrame(sensor_rows), use_container_width=True, hide_index=True)

        # ── Liegezeit-Plot
        st.markdown("")
        fig_heat = plot_wear_heatmap(wear_df, sensors)
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

        st.markdown("")
        fig_box = plot_wear_boxplot(wear_df)
        st.pyplot(fig_box, use_container_width=True)
        plt.close(fig_box)

        st.markdown("")
        fig_rank = plot_wear_ranking(wear_df, sensors)
        st.pyplot(fig_rank, use_container_width=True)
        plt.close(fig_rank)

        # ── Hinweis-Box
        max_wear = _max_wear
        long_sensors = [s for s in sensors if s["wear_days"] > max_wear * 0.9]
        warmup_note = f"Tag 1 (≈ {_warmup_h:.0f}h Einlaufphase) ist in der Vergleichsgrafik gesondert markiert (□)."

        st.markdown(f"""
        <div class="info-card">
        <strong>Interpretation:</strong><br>
        Sensor-Typ: <code>{source_name}</code> — max. Liegedauer: <strong>{max_wear} Tage</strong><br>
        {warmup_note}<br>
        Kompressionsartefakte nehmen typischerweise in den letzten Liegetagen zu
        (Gewebsreaktion, Membranermüdung). Ein ↑-Pfeil kennzeichnet eine signifikante Zunahme.<br>
        {"⚠ <strong>Hinweis:</strong> Mindestens ein Sensor nähert sich der maximalen Liegedauer." if long_sensors else ""}
        </div>
        """, unsafe_allow_html=True)

        # ── Tagesdetail-Tabelle (aufklappbar)
        with st.expander("Rohdaten: Fehlerrate pro Liegetag"):
            disp = wear_df.copy()
            disp["Sensor"] = disp["sensor_id"].apply(lambda x: f"#{x}")
            disp["Einlaufphase"] = disp["is_warmup"].map({True: "✓", False: ""})
            disp = disp.rename(columns={
                "day": "Liegetag", "n": "Messungen",
                "roc_pct": "RoC %", "comp_pct": "Kompr. %",
                "plat_pct": "Plateau %", "night_comp_pct": "Nacht-Kompr. %",
                "date_label": "Datum"
            })
            cols_show = ["Sensor", "Liegetag", "Datum", "Einlaufphase",
                         "Messungen", "RoC %", "Kompr. %", "Plateau %", "Nacht-Kompr. %"]
            st.dataframe(
                disp[cols_show].style.format({
                    "RoC %": "{:.2f}", "Kompr. %": "{:.2f}",
                    "Plateau %": "{:.2f}", "Nacht-Kompr. %": "{:.1f}",
                }),
                use_container_width=True, hide_index=True
            )

with tab6:
    st.markdown("#### 📄 Report")

    # Ermittle ob Liegezeit-Daten vorhanden
    _wear_df_rep  = wear_df  if "wear_df"  in dir() else None
    _sensors_rep  = sensors  if "sensors"  in dir() else None
    # wear_df wird im Liegezeit-Tab berechnet — sicher abfragen
    try:
        _wear_df_rep = wear_df
        _sensors_rep = sensors
    except NameError:
        _wear_df_rep = None
        _sensors_rep = None

    with st.spinner("Report wird erstellt…"):
        pdf_bytes = generate_pdf_report(
            df, meta, score, len(gaps_df),
            wear_df=_wear_df_rep, sensors=_sensors_rep
        )

    if pdf_bytes:
        # ── Inline-Anzeige im Browser
        import base64
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_html = f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%" height="900px"
            style="border: 1px solid #CBD5E0; border-radius: 6px;">
        </iframe>
        """
        st.markdown(pdf_html, unsafe_allow_html=True)

        # ── Download-Button darunter
        st.download_button(
            label="⬇ PDF herunterladen",
            data=pdf_bytes,
            file_name=f"cgm_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
    else:
        st.markdown("""
        <div class="warn-card">
        ⚠ <strong>reportlab nicht installiert.</strong><br>
        Einmalig ausführen: <code>pip install reportlab</code>
        </div>
        """, unsafe_allow_html=True)


