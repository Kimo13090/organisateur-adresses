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
uploaded_file = st.file_uploader(
    "Chargez votre fichier Excel contenant les adresses", type=["xlsx"]
)
if not uploaded_file:
    st.stop()
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Impossible de lire le fichier Excel : {e}")
    st.stop()

# --- Sélection des colonnes ---
st.sidebar.subheader("Paramétrage des colonnes")
cols = df.columns.tolist()
address_col = st.sidebar.selectbox("Colonne adresse", cols)
postal_col = st.sidebar.selectbox("Colonne code postal", cols)
city_col = st.sidebar.selectbox("Colonne ville", cols)
if not all([address_col, postal_col, city_col]):
    st.error("Sélectionnez les colonnes Adresse, Code postal et Ville.")
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
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(a) for a in df['Adresse complète']]
df['Latitude'], df['Longitude'] = zip(*coords)

# --- Filtrer adresses valides ---
df = df.dropna(subset=['Latitude','Longitude']).reset_index(drop=True)
if df.empty:
    st.error("Aucune adresse n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df)} adresses géocodées.")

# --- Filtrer secteur hors-distance ---
# Calculer centroid des points
centroid = (df['Latitude'].mean(), df['Longitude'].mean())
# Distance au centroid
df['dist_centroid'] = df.apply(
    lambda r: geodesic((r['Latitude'],r['Longitude']), centroid).meters, axis=1
)
# Séparer hors secteur >5km
df_in = df[df['dist_centroid'] <= 5000].reset_index(drop=True)
df_out = df[df['dist_centroid'] > 5000].reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur (>5km du centre) :")
    st.dataframe(df_out[[address_col,postal_col,city_col,'dist_centroid']])
st.info(f"{len(df_in)} adresses dans le secteur.")
if df_in.empty:
    st.stop()

# --- Choix du départ automatique : point le plus proche du centroid ---
df_in['dist0'] = df_in.apply(
    lambda r: geodesic((r['Latitude'],r['Longitude']), centroid).meters, axis=1
)
start_idx = df_in['dist0'].idxmin()
# Réordonner en plaçant ce point en tête
df_ordered = pd.concat(
    [df_in.loc[[start_idx]], df_in.drop(start_idx)], ignore_index=True
)

# --- Optimisation de l'ordre avec OSRM Trip ---
with st.spinner("Optimisation de l'ordre de tournée via OSRM Trip..."):
    try:
        wpts = ";".join(f"{lon},{lat}" for lat, lon in zip(df_ordered['Latitude'], df_ordered['Longitude']))
        trip_url = (
            f"http://router.project-osrm.org/trip/v1/driving/{wpts}?"
            "source=first&roundtrip=false&overview=full&geometries=geojson"
        )
        res = requests.get(trip_url, timeout=15)
        res.raise_for_status()
        waypoints_order = res.json()['trips'][0]['waypoint_order']
        order = [0] + [i+1 for i in waypoints_order]
    except:
        # Fallback tri angulaire autour du départ
        import math
        lat0, lon0 = df_ordered.loc[0, ['Latitude','Longitude']]
        df_tmp = df_ordered.copy()
        df_tmp['angle'] = df_tmp.apply(
            lambda r: math.atan2(r['Longitude']-lon0, r['Latitude']-lat0), axis=1
        )
        order = df_tmp.sort_values('angle').index.tolist()
# DataFrame optimisé
df_opt = df_ordered.iloc[order].reset_index(drop=True)

# --- Affichage du tableau ---
st.subheader("Tournée optimisée")
st.dataframe(df_opt[[address_col,postal_col,city_col,'Latitude','Longitude']], use_container_width=True)

# --- Itinéraire réel et instructions ---
with st.spinner("Génération de l'itinéraire..."):
    try:
        wpts2 = ";".join(f"{lon},{lat}" for lat, lon in zip(df_opt['Latitude'], df_opt['Longitude']))
        url2 = (
            f"http://router.project-osrm.org/route/v1/driving/{wpts2}?"
            "overview=full&geometries=geojson&steps=true"
        )
        r2 = requests.get(url2, timeout=20)
        r2.raise_for_status()
        data2 = r2.json()['routes'][0]
        coords2 = [(lat, lon) for lon, lat in data2['geometry']['coordinates']]
        instructions = []
        for leg in data2['legs']:
            for step in leg['steps']:
                m = step['maneuver']
                t = m.get('type','').capitalize()
                mod = m.get('modifier','')
                name = step.get('name','')
                d = int(step.get('distance',0))
                instructions.append(f"{t} {mod} sur {name} - {d} m")
    except:
        coords2 = list(zip(df_opt['Latitude'], df_opt['Longitude']))
        instructions = []

# --- Carte interactive ---
m = folium.Map(location=coords2[0], zoom_start=12)
poly = folium.PolyLine(coords2, color='blue', weight=4, opacity=0.7)
m.add_child(poly)
PolyLineTextPath(poly, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})
for idx,(lat,lon) in enumerate(zip(df_opt['Latitude'],df_opt['Longitude'])):
    color='green' if idx==0 else ('red' if idx==len(df_opt)-1 else 'blue')
    html=f"<div style='background:{color};color:white;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:bold'>{idx+1}</div>"
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html), tooltip=f"Étape {idx+1}: {df_opt[address_col].iloc[idx]}").add_to(m)
st.subheader("Itinéraire interactif")
st_folium(m,width=800,height=600)

# --- Instructions ---
if instructions:
    st.subheader("Instructions de conduite")
    for i,ins in enumerate(instructions,1): st.markdown(f"**{i}.** {ins}")

# --- Export XLSX ---
out=io.BytesIO()
with pd.ExcelWriter(out,engine='openpyxl') as w: df_opt.to_excel(w,index=False,sheet_name='Repérage')
out.seek(0)
st.download_button(
    label="Télécharger la tournée (.xlsx)",
    data=out,
    file_name='tournee_organisee.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
