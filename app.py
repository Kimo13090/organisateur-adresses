import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import requests
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# Configuration de la page
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Organisateur d'adresses pour repérages")

# --- Téléversement du fichier Excel ---
uploaded_file = st.file_uploader("Chargez votre fichier Excel avec adresses", type=["xlsx"])
if not uploaded_file:
    st.stop()
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Lecture du fichier impossible : {e}")
    st.stop()

# --- Sélection des colonnes ---
st.sidebar.subheader("Colonnes d'adresse")
cols = df.columns.tolist()
address_col = st.sidebar.selectbox("Adresse", cols)
postal_col = st.sidebar.selectbox("Code postal", cols)
city_col = st.sidebar.selectbox("Ville", cols)
if not all([address_col, postal_col, city_col]):
    st.error("Veuillez choisir les colonnes Adresse, Code postal et Ville.")
    st.stop()

# --- Construction des adresses complètes ---
df['Adresse complète'] = (
    df[address_col].astype(str) + ", " +
    df[postal_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# --- Géocodage (cache) ---
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

with st.spinner("Géocodage en cours..."):
    coords = [geocode(a) for a in df['Adresse complète']]
df['Latitude'], df['Longitude'] = zip(*coords)
df = df.dropna(subset=['Latitude','Longitude']).reset_index(drop=True)
if df.empty:
    st.error("Aucune adresse valide.")
    st.stop()
st.success(f"{len(df)} adresses géocodées.")

# --- Filtrer hors secteur (>5km du centroïde) ---
centroid = (df['Latitude'].mean(), df['Longitude'].mean())
df['dist_centroid'] = df.apply(lambda r: geodesic((r['Latitude'],r['Longitude']), centroid).meters, axis=1)
df_in = df[df['dist_centroid'] <= 5000].reset_index(drop=True)
df_out = df[df['dist_centroid'] > 5000].reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur (>5km du centre) :")
    st.dataframe(df_out[[address_col,postal_col,city_col,'dist_centroid']])
if df_in.empty:
    st.error("Aucune adresse dans le secteur.")
    st.stop()

# --- Choix du point de départ : plus proche voisin ---
# Calcul de la distance au plus proche autre adresse pour chaque point
def min_neighbor_distance(df_points, idx):
    origin = (df_points.loc[idx,'Latitude'], df_points.loc[idx,'Longitude'])
    others = df_points.drop(idx)
    dists = others.apply(lambda r: geodesic(origin, (r['Latitude'],r['Longitude'])).meters, axis=1)
    return dists.min()

nn_dists = [min_neighbor_distance(df_in, i) for i in range(len(df_in))]
start_idx = int(pd.Series(nn_dists).idxmin())
# Réordonner en plaçant ce point en tête
df_ordered = pd.concat([df_in.loc[[start_idx]], df_in.drop(start_idx)], ignore_index=True)

# --- Optimisation de l'ordre via OSRM Trip API ---
with st.spinner("Optimisation de la tournée..."):
    try:
        wpts = ";".join(f"{lon},{lat}" for lat, lon in zip(df_ordered['Latitude'], df_ordered['Longitude']))
        trip_url = f"http://router.project-osrm.org/trip/v1/driving/{wpts}?source=first&roundtrip=false&overview=full&geometries=geojson"
        res = requests.get(trip_url, timeout=15); res.raise_for_status()
        waypoints = res.json()['trips'][0]['waypoint_order']
        order = [0] + [i+1 for i in waypoints]
    except:
        # fallback tri angulaire autour du départ
        import math
        lat0, lon0 = df_ordered.loc[0,['Latitude','Longitude']]
        df_tmp = df_ordered.copy()
        df_tmp['angle'] = df_tmp.apply(
            lambda r: math.atan2(r['Longitude']-lon0, r['Latitude']-lat0), axis=1
        )
        order = df_tmp.sort_values('angle').index.tolist()

df_opt = df_ordered.iloc[order].reset_index(drop=True)

# --- Affichage du tableau optimisé ---
st.subheader("Tournée optimisée")
st.dataframe(df_opt[[address_col,postal_col,city_col,'Latitude','Longitude']], use_container_width=True)

# --- Génération de l'itinéraire et instructions ---
with st.spinner("Génération de l'itinéraire..."):
    try:
        w2 = ";".join(f"{lon},{lat}" for lat, lon in zip(df_opt['Latitude'], df_opt['Longitude']))
        url2 = f"http://router.project-osrm.org/route/v1/driving/{w2}?overview=full&geometries=geojson&steps=true"
        r2 = requests.get(url2, timeout=20); r2.raise_for_status()
        route = r2.json()['routes'][0]
        coords_route = [(lat,lon) for lon,lat in route['geometry']['coordinates']]
        instructions = []
        for leg in route['legs']:
            for stp in leg['steps']:
                m = stp['maneuver']
                t = m.get('type','').capitalize(); d = m.get('modifier','')
                nm = stp.get('name',''); dist = int(stp.get('distance',0))
                instructions.append(f"{t} {d} sur {nm} - {dist} m")
    except:
        coords_route = list(zip(df_opt['Latitude'], df_opt['Longitude'])); instructions = []

# --- Carte interactive ---
m = folium.Map(location=coords_route[0], zoom_start=12)
poly = folium.PolyLine(coords_route, color='blue', weight=4, opacity=0.7); m.add_child(poly)
PolyLineTextPath(poly, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})
for i,(lat,lon) in enumerate(zip(df_opt['Latitude'],df_opt['Longitude'])):
    col = 'green' if i==0 else ('red' if i==len(df_opt)-1 else 'blue')
    html = f"<div style='background:{col};color:white;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:bold'>{i+1}</div>"
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html), tooltip=f"Étape {i+1}: {df_opt[address_col].iloc[i]}").add_to(m)
st.subheader("Itinéraire interactif")
st_folium(m, width=800, height=600)

# --- Instructions ---
if instructions:
    st.subheader("Instructions de conduite")
    for idx, txt in enumerate(instructions,1): st.markdown(f"**{idx}.** {txt}")

# --- Export XLSX ---
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as w: df_opt.to_excel(w, index=False, sheet_name='Repérage')
out.seek(0)
st.download_button(label="Télécharger la tournée (.xlsx)", data=out, file_name='tournee_organisee.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
