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
df = pd.read_excel(uploaded_file)

# Sélection dynamique des colonnes
st.sidebar.subheader("Configuration des colonnes")
cols = df.columns.tolist()
address_col = st.sidebar.selectbox("Colonne adresse", cols)
postal_col = st.sidebar.selectbox("Colonne code postal", cols)
city_col = st.sidebar.selectbox("Colonne ville", cols)

# Vérifier la sélection
if not address_col or not postal_col or not city_col:
    st.error("Veuillez sélectionner toutes les colonnes nécessaires.")
    st.stop()

# Construction de l'adresse complète
df["Adresse complète"] = (
    df[address_col].astype(str) + ", " +
    df[postal_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# Géocodeur OSM avec cache
geolocator = Nominatim(user_agent="repérage_web_app")
@st.cache_data
def geocode(addr):
    try:
        loc = geolocator.geocode(addr, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

# Géocodage des adresses
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(addr) for addr in df["Adresse complète"]]
# Extraire latitudes et longitudes
lats, lons = zip(*coords)
df["Latitude"] = lats
df["Longitude"] = lons

# Filtrer les adresses non géocodées
df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées avec succès.")

# Optimisation via OSRM Trip API
with st.spinner("Optimisation de la tournée (OSRM Trip)..."):
    waypoints = ";".join(
        f"{lon},{lat}" for lat, lon in zip(df_clean["Latitude"], df_clean["Longitude"])
    )
    trip_url = (
        f"http://router.project-osrm.org/trip/v1/driving/{waypoints}"
        f"?source=first&roundtrip=false&overview=full&geometries=geojson"
    )
    try:
        response = requests.get(trip_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        trip = data["trips"][0]
        order = trip.get("waypoint_order", list(range(len(df_clean))))
        coords_geo = trip["geometry"]["coordinates"]
        route_pts = [(lat, lon) for lon, lat in coords_geo]
        df_opt = df_clean.iloc[order].reset_index(drop=True)
    except Exception:
        # Fallback Nearest Neighbor
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
        df_opt = nearest_neighbor(df_clean)
        route_pts = list(zip(df_opt["Latitude"], df_opt["Longitude"]))

# Affichage des adresses optimisées
st.subheader("Adresses organisées pour la tournée")
st.dataframe(
    df_opt[[address_col, postal_col, city_col, "Latitude", "Longitude"]],
    use_container_width=True
)

# Création de la carte interactive
@st.cache_data
def create_map(route, markers, labels):
    m = folium.Map(location=markers[0], zoom_start=12)
    line = folium.PolyLine(route, color="blue", weight=4, opacity=0.7)
    m.add_child(line)
    PolyLineTextPath(line, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})
    for i, (lat, lon) in enumerate(markers):
        col = 'green' if i == 0 else ('red' if i == len(markers) - 1 else 'blue')
        html = (
            f"<div style='background:{col};color:white;border-radius:50%;width:28px;height:28px;"
            f"display:flex;align-items:center;justify-content:center;font-weight:bold'>"
            f"{i+1}</div>"
        )
        folium.Marker(
            location=(lat, lon), icon=folium.DivIcon(html=html),
            tooltip=f"Étape {i+1}: {labels[i]}", popup=labels[i]
        ).add_to(m)
    return m

map_obj = create_map(
    route_pts,
    list(zip(df_opt["Latitude"], df_opt["Longitude"])),
    df_opt[address_col].tolist()
)
st.subheader("Visualisation interactive de la tournée (véhicule)")
st_folium(map_obj, width=800, height=600)

# Exportation du fichier Excel
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
