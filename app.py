# --- Import Libraries ---
import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import io
import numpy as np
from datetime import datetime

# -----------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE -----------------------------------------------------
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Optimisateur de Tournées Logistiques",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# FEUILLE DE STYLE -------------------------------------------------------------
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #f5f7fa; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding:20px; border-radius:10px; text-align:center; color:#fff; }
        div[data-testid="metric-container"] { background:#fff; border:1px solid #e0e0e0; padding:1rem; border-radius:8px; }
        .stButton>button { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:0.75rem 2rem; border-radius:8px; }
        .download { background:linear-gradient(135deg,#48bb78,#38a169); color:#fff; padding:1.5rem; border-radius:10px; text-align:center; }
        .footer { background:#2d3748; color:#fff; padding:1rem; text-align:center; margin-top:2rem; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header">
        <h1>🚛 Optimisateur de Tournées Logistiques</h1>
        <p>Optimisez gratuitement vos itinéraires – jusqu'à 250 adresses</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# INITIALISATION DU SESSION STATE ---------------------------------------------
# -----------------------------------------------------------------------------
def _init_state():
    """Crée les clés du session_state si elles n'existent pas."""
    defaults = [
        "processed",
        "df_opt",
        "df_out",
        "df_fail",
        "total_dist",
        "excel_buf",
        "map_obj",
        "last_uploaded_name",
    ]
    for k in defaults:
        if k not in st.session_state:
            st.session_state[k] = None

_init_state()

# -----------------------------------------------------------------------------
# DÉTECTION INTELLIGENTE DES COLONNES -----------------------------------------
# -----------------------------------------------------------------------------

def smart_detect_cols(df: pd.DataFrame):
    """Trouve automatiquement les colonnes adresse / CP / ville."""
    addr = next((c for c in df.columns if any(x in c.lower() for x in ["adresse", "address", "rue", "voie"])), None)
    postal = next((c for c in df.columns if any(x in c.lower() for x in ["cp", "zip", "postal", "code"])), None)
    city = next((c for c in df.columns if any(x in c.lower() for x in ["ville", "city", "commune"])), None)
    return addr, postal, city

# -----------------------------------------------------------------------------
# OUTILS GÉOCODAGE -------------------------------------------------------------
# -----------------------------------------------------------------------------

# ⚠️  Correction : st.cache_data ne peut pas sérialiser RateLimiter.
#                 On passe à st.cache_resource (objet mutable autorisé).

@st.cache_resource(show_spinner=False)
def get_geocode_function():
    geolocator = Nominatim(user_agent="logistics_app_v4")
    return RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)


def batch_geocode(addresses: pd.Series, limit: int = 250) -> pd.DataFrame:
    """Retourne un DataFrame lat/lon pour chaque adresse (NaN si échec)."""
    geocode = get_geocode_function()
    latlons = []
    progress = st.progress(0.0, text="Géocodage…")
    for i, adr in enumerate(addresses[:limit]):
        loc = geocode(adr)
        latlons.append((loc.latitude, loc.longitude) if loc else (np.nan, np.nan))
        progress.progress((i + 1) / min(len(addresses), limit))
    progress.empty()
    return pd.DataFrame(latlons, columns=["lat", "lon"])

# -----------------------------------------------------------------------------
# DÉTECTION DES ADRESSES HORS SECTEUR -----------------------------------------
# -----------------------------------------------------------------------------

def detect_outliers(df_geo: pd.DataFrame):
    if len(df_geo) < 5:
        return list(df_geo.index), []
    center = (df_geo["lat"].median(), df_geo["lon"].median())
    df_geo["dist_centre"] = df_geo.apply(lambda r: geodesic(center, (r.lat, r.lon)).km, axis=1)
    med = np.median(df_geo["dist_centre"])
    mad = np.median(np.abs(df_geo["dist_centre"] - med))
    seuil = max(med + 2.5 * mad, 10)  # >=10 km mini
    in_idx = df_geo[df_geo["dist_centre"] <= seuil].index.tolist()
    out_idx = df_geo[df_geo["dist_centre"] > seuil].index.tolist()
    return in_idx, out_idx

# -----------------------------------------------------------------------------
# ALGO D'OPTIMISATION (Nearest‑Neighbor + 2‑Opt) -------------------------------
# -----------------------------------------------------------------------------

def optimise_itineraire(df_pts: pd.DataFrame):
    pts = df_pts.copy().reset_index()
    centre = pts[["lat", "lon"]].mean().tolist()
    start = (
        pts.assign(dist=lambda d: d.apply(lambda r: geodesic(centre, (r.lat, r.lon)).km, axis=1))
        .nsmallest(1, "dist")
        .index[0]
    )
    current = start
    unvisited = set(pts.index) - {current}
    route = [current]
    while unvisited:
        nxt = min(
            unvisited,
            key=lambda i: geodesic(
                (pts.loc[current, "lat"], pts.loc[current, "lon"]),
                (pts.loc[i, "lat"], pts.loc[i, "lon"]),
            ).km,
        )
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    # 2‑opt pour améliorer le circuit
    def longueur(tour):
        return sum(
            geodesic((pts.loc[a, "lat"], pts.loc[a, "lon"]), (pts.loc[b, "lat"], pts.loc[b, "lon"])).km
            for a, b in zip(tour, tour[1:])
        )

    best = route
    best_len = longueur(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                new = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                nl = longueur(new)
                if nl < best_len:
                    best, best_len, improved = new, nl, True
    df_route = pts.loc[best].reset_index(drop=True)
    return df_route, best_len

# -----------------------------------------------------------------------------
# DISTANCES CUMULÉES -----------------------------------------------------------
# -----------------------------------------------------------------------------

def add_distances(df_route: pd.DataFrame):
    d = [0]
    for i in range(1, len(df_route)):
        d.append(
            geodesic(
                (df_route.loc[i - 1, "lat"], df_route.loc[i - 1, "lon"]),
                (df_route.loc[i, "lat"], df_route.loc[i, "lon"]),
            ).km
        )
    df_route["dist_etape"] = d
    df_route["dist_cumulee"] = df_route["dist_etape"].cumsum()
    return df_route

# -----------------------------------------------------------------------------
# CARTE FOLIUM -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_map(df_rt: pd.DataFrame, df_out: pd.DataFrame):
    carte = folium.Map(location=[df_rt.lat.mean(), df_rt.lon.mean()], zoom_start=12)
    for idx, row in df_rt.iterrows():
        icon = "play" if idx == 0 else "stop" if idx == len(df_rt) - 1 else "location-dot"
        colour = "green" if idx == 0 else "red" if idx == len(df_rt) - 1 else "blue"
        folium.Marker(
            [row.lat, row.lon],
            popup=f"<b>{idx+1}</b> {row.address}<br>{row.dist_etape:.1f} km",
            icon=folium.Icon(color=colour, icon=icon, prefix="fa"),
        ).add_to(carte)
    folium.PolyLine(df_rt[["lat", "lon"]].values.tolist(), color="purple", weight=3).add_to(carte)
    if not df_out.empty:
        for _, r in df_out.iterrows():
            folium.Marker(
                [r.lat, r.lon],
                popup=r.address,
                icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
            ).add_to(carte)
    return carte

# -----------------------------------------------------------------------------
# EXPORT EXCEL -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def make_excel(df_rt: pd.DataFrame, df_out: pd.DataFrame, df_fail: pd.DataFrame, total_km: float):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_rt.assign(Ordre=range(1, len(df_rt) + 1)).to_excel(
            writer, index=False, sheet_name="Itineraire"
        )
        if not df_out.empty:
            df_out.to_excel(writer, index=False, sheet_name="Hors_Secteur")
        if not df_fail.empty:
            df_fail.to_excel(writer, index=False, sheet_name="Echecs")
        synth = pd.DataFrame(
            {
                "Métrique": ["Livraisons optimisées", "Hors secteur", "Échecs geocoding", "Distance totale (km)"]
