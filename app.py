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
    st.error(f"Impossible de lire le fichier Excel : {e}")
    st.stop()

# Sidebar : configuration des colonnes
st.sidebar.subheader("Configuration des colonnes d'adresse")
cols = df.columns.tolist()
address_col = st.sidebar.selectbox("Colonne adresse", cols)
postal_col = st.sidebar.selectbox("Colonne code postal", cols)
city_col = st.sidebar.selectbox("Colonne ville", cols)

if not all([address_col, postal_col, city_col]):
    st.error("Merci de sélectionner les 3 colonnes nécessaires.")
    st.stop()

# Construction de l'adresse complète
df["Adresse complète"] = (
    df[address_col].astype(str) + ", " +
    df[postal_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# Géocodeur OpenStreetMap avec cache
geolocator = Nominatim(user_agent="rep_app")
@st.cache_data
# Géocode une seule fois chaque adresse
def geocode(addr):
    try:
        loc = geolocator.geocode(addr, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

# Géocodage de toutes les adresses
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(a) for a in df["Adresse complète"]]

# Extraction des coordonnées
lats, lons = zip(*coords)
df["Latitude"], df["Longitude"] = lats, lons

# Filtrer uniquement les adresses valides
df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées avec succès.")

# Calcul de l'ordre de la tournée via OSRM Trip API ou fallback NN
with st.spinner("Optimisation de la tournée..."):
    try:
        waypoints = ";".join(f"{lon},{lat}" for lat, lon in zip(df_clean["Latitude"], df_clean["Longitude"]))
        trip_url = (
            f"http://router.project-osrm.org/trip/v1/driving/{waypoints}" +
            "?source=first&roundtrip=false&overview=full&geometries=geojson"
        )
        r = requests.get(trip_url, timeout=15)
        r.raise_for_status()
        data = r.json()
        trip = data["trips"][0]
        order = trip.get("waypoint_order", list(range(len(df_clean))))
    except Exception:
        # Fallback nearest neighbor si l'API Trip échoue
        def nearest_neighbor(df_nn):
            seq = [df_nn.iloc[0]]
            rem = df_nn.iloc[1:].copy().reset_index(drop=True)
            while not rem.empty:
                last = seq[-1]
                origin = (last["Latitude"], last["Longitude"])
                dists = rem.apply(
                    lambda r: geodesic(origin, (r["Latitude"], r["Longitude"])).meters,
                    axis=1
                )
                idx_min = dists.idxmin()
                seq.append(rem.loc[idx_min])
                rem = rem.drop(idx_min).reset_index(drop=True)
            return pd.DataFrame(seq)
        df_seq = nearest_neighbor(df_clean)
        order = df_seq.index.tolist()

# DataFrame ordonné
df_opt = df_clean.iloc[order].reset_index(drop=True)

# Affichage du tableau optimisé
st.subheader("Adresses organisées pour la tournée")
st.dataframe(
    df_opt[[address_col, postal_col, city_col, "Latitude", "Longitude"]],
    use_container_width=True
)

# Construction de la route segment par segment (OSRM Route API)
@st.cache_data
# Construit une liste de points lat, lon suivant la route réelle
def build_route(pts):
    route = []
    for start, end in zip(pts[:-1], pts[1:]):
        lon1, lat1 = start[1], start[0]
        lon2, lat2 = end[1], end[0]
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        )
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            coords = res.json()["routes"][0]["geometry"]["coordinates"]
            route.extend([(lat, lon) for lon, lat in coords])
        except:
            # Si échec, tracer droit
            route.append(start)
            route.append(end)
    return route

coords_list = list(zip(df_opt["Latitude"], df_opt["Longitude"]))
route_pts = build_route(coords_list)

# Création de la carte interactive avec marqueurs numérotés
m = folium.Map(location=coords_list[0], zoom_start=12)
line = folium.PolyLine(route_pts, color="blue", weight=4, opacity=0.7)
m.add_child(line)
# Flèches le long de la route
PolyLineTextPath(line, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})

# Ajout des marqueurs
for idx, (lat, lon) in enumerate(coords_list):
    # Couleurs : début vert, fin rouge, intermédiaire bleu
    color = 'green' if idx == 0 else ('red' if idx == len(coords_list)-1 else 'blue')
    html = f"<div style='background:{color};color:white;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:bold'>{idx+1}</div>"
    folium.Marker(
        location=(lat, lon),
        icon=folium.DivIcon(html=html),
        tooltip=f"Étape {idx+1}: {df_opt[address_col].iloc[idx]}"
    ).add_to(m)
st.subheader("Itinéraire interactif (voiture)")
st_folium(m, width=800, height=600)

# Export du fichier Excel
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
