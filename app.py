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

# Téléversement du fichier Excel
uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])
if not uploaded_file:
    st.stop()
# Lecture du fichier
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Impossible de lire le fichier Excel : {e}")
    st.stop()

# Sélection dynamique des colonnes
st.sidebar.subheader("Paramètres des colonnes")
cols = df.columns.tolist()
address_col = st.sidebar.selectbox("Colonne adresse", cols)
postal_col = st.sidebar.selectbox("Colonne code postal", cols)
city_col = st.sidebar.selectbox("Colonne ville", cols)
if not all([address_col, postal_col, city_col]):
    st.error("Sélectionnez les colonnes Adresse, Code postal et Ville.")
    st.stop()

# Construction des adresses complètes
df['Adresse complète'] = (
    df[address_col].astype(str) + ", " +
    df[postal_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# Géocodage via Nominatim (cache)
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
# Exécution du géocodage
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(a) for a in df['Adresse complète']]
# Affectation des coordonnées
df['Latitude'], df['Longitude'] = zip(*coords)
# Filtrer adresses valides
df_clean = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse valide n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées.")

# Détection des adresses hors secteur (>5km du premier point)
origin = (df_clean.loc[0, 'Latitude'], df_clean.loc[0, 'Longitude'])
df_clean['distance_origin'] = df_clean.apply(
    lambda r: geodesic(origin, (r['Latitude'], r['Longitude'])).meters,
    axis=1
)
threshold = 5000  # mètres
df_in = df_clean[df_clean['distance_origin'] <= threshold].reset_index(drop=True)
df_out = df_clean[df_clean['distance_origin'] > threshold].reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur (>5km) :")
    st.dataframe(df_out[[address_col, postal_col, city_col, 'distance_origin']])
st.info(f"{len(df_in)} adresses dans le secteur (<= {threshold} m).")
if df_in.empty:
    st.stop()

# --- Choix du point de départ ---
st.sidebar.subheader("Point de départ de la tournée")
start_address = st.sidebar.selectbox(
    "Sélectionnez le point de départ", df_in[address_col].tolist(), index=0
)
# Réordonner df_in pour placer le point de départ en premier
df_start = df_in[df_in[address_col] == start_address]
df_rest = df_in[df_in[address_col] != start_address]
df_ordered = pd.concat([df_start, df_rest], ignore_index=True)

# Optimisation de l'ordre via OSRM Trip API (source fixé au premier point)
with st.spinner("Optimisation de l'ordre de tournée via OSRM Trip..."):
    try:
        wpts = ";".join(
            f"{lon},{lat}" for lat, lon in zip(df_ordered['Latitude'], df_ordered['Longitude'])
        )
        # source=first force le départ au premier waypoint\ n        trip_url = f"http://router.project-osrm.org/trip/v1/driving/{wpts}?source=first&roundtrip=false&overview=full&geometries=geojson"
        res = requests.get(trip_url, timeout=15)
        res.raise_for_status()
        data = res.json()
        # OSRM Trip renvoie l'ordre des waypoints hors point de départ, on l'insère en tête
        waypoint_order = data['trips'][0].get('waypoint_order', list(range(1, len(df_ordered))))
        order = [0] + [i+1 for i in waypoint_order]
    except Exception:
        # Fallback tri angulaire autour du départ
        import math
        origin_lat, origin_lon = df_ordered.loc[0, ['Latitude','Longitude']]
        df_tmp = df_ordered.copy()
        df_tmp['angle'] = df_tmp.apply(
            lambda r: math.atan2(r['Longitude']-origin_lon, r['Latitude']-origin_lat), axis=1
        )
        df_tmp = df_tmp.sort_values('angle')
        # On replace le point de départ en tête
        order = [df_tmp.index.get_loc(0)] + [i for i in df_tmp.index if i != 0]

# DataFrame ordonné final
df_opt = df_ordered.iloc[order].reset_index(drop=True)
df_opt = df_in.iloc[order].reset_index(drop=True)

# Affichage du tableau optimisé
st.subheader("Adresses optimisées pour la tournée")
st.dataframe(
    df_opt[[address_col, postal_col, city_col, 'Latitude', 'Longitude']], use_container_width=True
)

# Génération de la route et instructions via OSRM Route API
with st.spinner("Génération d'itinéraire..."):
    try:
        wpts2 = ";".join(f"{lon},{lat}" for lat, lon in zip(df_opt['Latitude'], df_opt['Longitude']))
        route_url = (f"http://router.project-osrm.org/route/v1/driving/{wpts2}?overview=full&geometries=geojson&steps=true")
        r2 = requests.get(route_url, timeout=20)
        r2.raise_for_status()
        route_data = r2.json()['routes'][0]
        coords_geo2 = route_data['geometry']['coordinates']
        route_pts = [(lat, lon) for lon, lat in coords_geo2]
        # Instructions détaillées
        instructions = []
        for leg in route_data.get('legs', []):
            for step in leg.get('steps', []):
                m = step['maneuver']
                instr = m.get('type','').capitalize()
                mod = m.get('modifier','')
                street = step.get('name','')
                dist = int(step.get('distance',0))
                instructions.append(f"{instr} {mod} sur {street} - {dist} m")
    except Exception:
        route_pts = list(zip(df_opt['Latitude'], df_opt['Longitude']))
        instructions = []

# Carte interactive
m = folium.Map(location=route_pts[0], zoom_start=12)
line = folium.PolyLine(route_pts, color='blue', weight=4, opacity=0.7)
m.add_child(line)
PolyLineTextPath(line, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})
for idx, (lat, lon) in enumerate(zip(df_opt['Latitude'], df_opt['Longitude'])):
    color = 'green' if idx==0 else ('red' if idx==len(df_opt)-1 else 'blue')
    icon_html = f"<div style='background:{color};color:white;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:bold'>{idx+1}</div>"
    folium.Marker(location=(lat,lon), icon=folium.DivIcon(html=icon_html), tooltip=f"Étape {idx+1}: {df_opt[address_col].iloc[idx]}").add_to(m)
st.subheader("Itinéraire interactif (véhicule)")
st_folium(m, width=800, height=600)

# Instructions de conduite
if instructions:
    st.subheader("Instructions de conduite")
    for i, ins in enumerate(instructions, 1):
        st.markdown(f"**{i}.** {ins}")

# Export du fichier Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    df_opt.to_excel(writer, index=False, sheet_name='Repérage')
output.seek(0)
# Téléchargement
st.download_button(
    label="Télécharger la tournée organisée (.xlsx)",
    data=output,
    file_name='tournee_organisee.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
