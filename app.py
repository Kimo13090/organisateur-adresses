import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import requests
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# --- Page config ---
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Organisateur d'adresses pour repérages")

# --- File upload ---
uploaded = st.file_uploader("Chargez un fichier Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()
try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Erreur lecture Excel : {e}")
    st.stop()

# --- Column selection ---
st.sidebar.header("Colonnes des données")
cols = df.columns.tolist()
ad_col = st.sidebar.selectbox("Colonne Adresse", cols)
pc_col = st.sidebar.selectbox("Colonne Code Postal", cols)
ct_col = st.sidebar.selectbox("Colonne Ville", cols)
if not all([ad_col, pc_col, ct_col]):
    st.error("Sélectionnez les colonnes avant de continuer.")
    st.stop()

# --- Build full address ---
df['full_addr'] = (
    df[ad_col].astype(str) + ", " +
    df[pc_col].astype(str) + " " +
    df[ct_col].astype(str) + ", France"
)

# --- Geocode with cache ---
geolocator = Nominatim(user_agent="rep_app")
@st.cache_data
def geocode(addr):
    try:
        loc = geolocator.geocode(addr, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

with st.spinner("Géocodage en cours... "):
    coords = [geocode(a) for a in df['full_addr']]
df['lat'], df['lon'] = zip(*coords)
df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
if df.empty:
    st.error("Aucune adresse géocodée.")
    st.stop()
st.success(f"Géocodé : {len(df)} adresses")

# --- Filter distant outliers (>5km from centroid) ---
center = (df['lat'].mean(), df['lon'].mean())
df['dist_center'] = df.apply(
    lambda r: geodesic((r['lat'],r['lon']), center).meters, axis=1
)
df_in = df[df['dist_center'] <= 5000].reset_index(drop=True)
df_out = df[df['dist_center'] > 5000].reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur (>5km)")
    st.dataframe(df_out[[ad_col,pc_col,ct_col,'dist_center']])
if df_in.empty:
    st.error("Aucune adresse dans le secteur.")
    st.stop()

# --- Nearest-Neighbor routing starting from most central point ---
# 1. Find starting index: minimize sum of distances to all others
sum_dists = df_in.apply(
    lambda r: df_in.apply(lambda x: geodesic((r['lat'],r['lon']), (x['lat'],x['lon'])).meters, axis=1).sum(),
    axis=1
)
start = sum_dists.idxmin()

# 2. Greedy nearest neighbor order
def greedy_order(df_pts, start_idx):
    visited = [start_idx]
    remaining = set(df_pts.index) - {start_idx}
    while remaining:
        last = visited[-1]
        dists = {i: geodesic((df_pts.at[last,'lat'],df_pts.at[last,'lon']),
                             (df_pts.at[i,'lat'],df_pts.at[i,'lon'])).meters
                 for i in remaining}
        nxt = min(dists, key=dists.get)
        visited.append(nxt)
        remaining.remove(nxt)
    return visited
orden = greedy_order(df_in, start)

df_ord = df_in.loc[orden].reset_index(drop=True)

# --- Display organized table ---
st.subheader("Tournée organisée")
st.dataframe(df_ord[[ad_col,pc_col,ct_col,'lat','lon']], use_container_width=True)

# --- Build route segments via OSRM Route API ---
with st.spinner("Construction de l'itinéraire..."):
    route_points = []
    for i in range(len(df_ord)-1):
        a = df_ord.loc[i, ['lon','lat']]
        b = df_ord.loc[i+1, ['lon','lat']]
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{a['lon']},{a['lat']};{b['lon']},{b['lat']}?overview=full&geometries=geojson")
        try:
            res = requests.get(url, timeout=10).json()
            seg = res['routes'][0]['geometry']['coordinates']
            route_points.extend([(lat,lon) for lon,lat in seg])
        except:
            # direct fallback
            route_points.append((df_ord.loc[i,'lat'],df_ord.loc[i,'lon']))
            route_points.append((df_ord.loc[i+1,'lat'],df_ord.loc[i+1,'lon']))

# --- Plot map ---
m = folium.Map(location=route_points[0], zoom_start=12)
line = folium.PolyLine(route_points, color='blue', weight=4, opacity=0.7)
m.add_child(line)
PolyLineTextPath(line, '▶', repeat=True, offset=10,
                 attributes={'fill':'blue','font-size':'12'})
for idx,(lat,lon) in enumerate(zip(df_ord['lat'],df_ord['lon'])):
    color = 'green' if idx==0 else ('red' if idx==len(df_ord)-1 else 'blue')
    icon = folium.DivIcon(html=f"<div style='background:{color};color:white;"
                                 "border-radius:50%;width:24px;height:24px;display:flex;"
                                 "align-items:center;justify-content:center;font-weight:bold'>"
                                 f"{idx+1}</div>")
    folium.Marker((lat,lon), icon=icon, tooltip=f"Étape {idx+1}: {df_ord.loc[idx,ad_col]}").add_to(m)
st.subheader("Carte interactive")
st_folium(m, width=800, height=600)

# --- Export XLSX ---
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    df_ord.to_excel(writer, index=False, sheet_name='Repérage')
out.seek(0)
st.download_button(label="Télécharger (.xlsx)", data=out,
                   file_name='tournee.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
