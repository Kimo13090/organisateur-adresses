import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import requests
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# --- Configuration de la page ---
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Organisateur d'adresses pour repérages")

# --- Téléversement du fichier Excel ---
uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])
if not uploaded_file:
    st.stop()
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Impossible de lire le fichier Excel : {e}")
    st.stop()

# --- Sélection dynamique des colonnes ---
st.sidebar.subheader("Paramètres des colonnes")
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

# --- Géocodage via Nominatim (cache) ---
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
# Exécuter le géocodage
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(a) for a in df['Adresse complète']]
# Assigner lat/lon
df['Latitude'], df['Longitude'] = zip(*coords)
# Filtrer les adresses valides
df_clean = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse valide n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées.")

# --- Filtrer secteur principal (distance du centre) ---
# Calculer le centre géographique
df_clean['dist_centre'] = df_clean.apply(
    lambda r: geodesic(
        (r['Latitude'], r['Longitude']),
        (df_clean['Latitude'].mean(), df_clean['Longitude'].mean())
    ).meters, axis=1
)
# Seuil en mètres (à ajuster si nécessaire)
sector_radius = st.sidebar.slider("Seuil distance au centre (m)", min_value=500, max_value=5000, value=2000)
mask = df_clean['dist_centre'] <= sector_radius
df_in = df_clean[mask].reset_index(drop=True)
df_out = df_clean[~mask].reset_index(drop=True)
if not df_out.empty:
    st.warning("Certaines adresses sont hors secteur :")
    st.dataframe(df_out[[address_col, postal_col, city_col]])
st.info(f"{len(df_in)} adresses dans le secteur (rayon {sector_radius}m).")
if df_in.empty:
    st.stop()

# --- Optimisation de l'ordre via OSRM Trip API ---
with st.spinner("Optimisation de l'ordre de tournée..."):
    try:
        wpts = ";".join(f"{lon},{lat}" for lat, lon in zip(df_in['Latitude'], df_in['Longitude']))
        trip_url = f"http://router.project-osrm.org/trip/v1/driving/{wpts}?source=first&roundtrip=false&overview=full&geometries=geojson"
        res = requests.get(trip_url, timeout=15)
        res.raise_for_status()
        data = res.json()
        order = data['trips'][0].get('waypoint_order', list(range(len(df_in))))
    except Exception:
        # Fallback tri angulaire si OSRM Trip échoue
        import math
        origin = df_in.iloc[0]
        df_temp = df_in.copy()
        df_temp['angle'] = df_temp.apply(
            lambda r: math.atan2(
                r['Longitude']-origin['Longitude'],
                r['Latitude']-origin['Latitude']
            ), axis=1
        )
        df_temp = df_temp.sort_values('angle')
        order = df_temp.index.tolist()
# Réordonner
df_opt = df_in.iloc[order].reset_index(drop=True)

# --- Affichage du tableau optimisé ---
st.subheader("Adresses optimisées pour tournées")
st.dataframe(df_opt[[address_col, postal_col, city_col, 'Latitude', 'Longitude']], use_container_width=True)

# --- Calcul de la route réelle et instructions ---
with st.spinner("Génération de l'itinéraire et instructions..."):
    try:
        wpts2 = ";".join(f"{lon},{lat}" for lat, lon in zip(df_opt['Latitude'], df_opt['Longitude']))
        route_url = (
            f"http://router.project-osrm.org/route/v1/driving/{wpts2}" +
            "?overview=full&geometries=geojson&steps=true"
        )
        r2 = requests.get(route_url, timeout=20)
        r2.raise_for_status()
        route_data = r2.json()['routes'][0]
        coords_geo2 = route_data['geometry']['coordinates']
        route_pts = [(lat, lon) for lon, lat in coords_geo2]
        # Instructions
        instructions = []
        for leg in route_data.get('legs', []):
            for step in leg.get('steps', []):
                m = step['maneuver']
                instr = m.get('type','').capitalize()
                mod = m.get('modifier','')
                street = step.get('name','')
                dist = step.get('distance',0)
                instructions.append(f"{instr} {mod} sur {street} ({int(dist)} m)")
    except Exception:
        route_pts = list(zip(df_opt['Latitude'], df_opt['Longitude']))
        instructions = []

# --- Carte interactive ---
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

# --- Affichage des instructions ---
if instructions:
    st.subheader("Instructions de conduite")
    for i, ins in enumerate(instructions,1): st.markdown(f"{i}. {ins}")

# --- Export Excel ---
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    df_opt.to_excel(writer, index=False, sheet_name='Repérage')
out.seek(0)
st.download_button("Télécharger la tournée organisée (.xlsx)", data=out, file_name='tournee_organisee.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df_opt.to_excel(writer, index=False, sheet_name="Repérage")
out.seek(0)

st.download_button(
    "Télécharger la tournée organisée (.xlsx)",
    data=out,
    file_name="tournee_organisee.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_opt.to_excel(writer, index=False, sheet_name="Repérage")
output.seek(0)

st.download_button(
    "Télécharger la tournée organisée (.xlsx)",
    data=output,
    file_name="tournee_organisee.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
