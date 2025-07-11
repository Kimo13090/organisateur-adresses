import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
import io
import requests
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# Configurer la page Streamlit
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Organisateur d'adresses pour repérages")

# Téléversement du fichier Excel
uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])  
if not uploaded_file:
    st.info("Veuillez importer un fichier Excel comportant les colonnes 'Adresse du client', 'CPSTCMN', 'LVIL'.")
    st.stop()

df = pd.read_excel(uploaded_file)
required_cols = ["Adresse du client", "CPSTCMN", "LVIL"]
if not all(col in df.columns for col in required_cols):
    missing = list(set(required_cols) - set(df.columns))
    st.error(f"Colonnes manquantes: {', '.join(missing)}")
    st.stop()

# Construire l'adresse complète
df["Adresse complète"] = (df["Adresse du client"] + ", " + df["CPSTCMN"].astype(str) + " " + df["LVIL"] + ", France")

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

# Géocodage
with st.spinner("Géocodage des adresses... (une seule fois) "):
    coords = [geocode(addr) for addr in df["Adresse complète"]]
lats, lons = zip(*coords)
df["Latitude"], df["Longitude"] = lats, lons

# Filtrer les adresses géocodées
df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse valide n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées avec succès.")

# Optimisation globale via OSRM Trip API
with st.spinner("Optimisation de la tournée (OSRM Trip)..."):
    # Préparer la chaîne de waypoints lon,lat;...
    waypoints = ";".join(f"{lon},{lat}" for lat, lon in zip(df_clean["Longitude"], df_clean["Latitude"]))
    trip_url = (
        f"http://router.project-osrm.org/trip/v1/driving/{waypoints}"
        f"?source=first&roundtrip=false&overview=full&geometries=geojson"
    )
    try:
        res = requests.get(trip_url, timeout=15)
        res.raise_for_status()
        data = res.json()
        if "trips" in data and data["trips"]:
            trip = data["trips"][0]
            order = trip.get("waypoint_order", list(range(len(df_clean))))
            coords_geo = trip["geometry"]["coordinates"]
            route_pts = [(lat, lon) for lon, lat in coords_geo]
            df_opt = df_clean.iloc[order].reset_index(drop=True)
        else:
            raise ValueError("Empty trips")
    except (requests.RequestException, ValueError, Exception) as e:
        st.warning("Impossible d'utiliser OSRM Trip, utilisation d'un itinéraire par plus proche voisin.")
        # Fallback Nearest Neighbor
        def nearest_neighbor(df_nn):
            seq = [df_nn.iloc[0]]
            reste = df_nn.iloc[1:].copy()
            while not reste.empty:
                last = seq[-1]
                origin = (last["Latitude"], last["Longitude"])
                dists = reste.apply(
                    lambda r: geodesic(origin, (r["Latitude"], r["Longitude"])) .meters,
                    axis=1
                )
                idx_min = dists.idxmin()
                seq.append(reste.loc[idx_min])
                reste = reste.drop(idx_min)
            return pd.DataFrame(seq)
        df_opt = nearest_neighbor(df_clean).reset_index(drop=True)
        route_pts = list(zip(df_opt["Latitude"], df_opt["Longitude"]))

# Affichage du tableau optimisé
st.subheader("Adresses organisées pour la tournée")
st.dataframe(
    df_opt[["Adresse du client", "CPSTCMN", "LVIL", "Latitude", "Longitude"]], use_container_width=True
)
st.subheader("Adresses organisées pour la tournée")
st.dataframe(
    df_opt[["Adresse du client", "CPSTCMN", "LVIL", "Latitude", "Longitude"]], use_container_width=True
)

# Création de la carte interactive
@st.cache_data
def create_map(route_pts, marker_pts, labels):
    m = folium.Map(location=marker_pts[0], zoom_start=12)
    # Tracé de la route
    line = folium.PolyLine(route_pts, color="blue", weight=4, opacity=0.7)
    m.add_child(line)
    PolyLineTextPath(line, '▶', repeat=True, offset=10,
                     attributes={'fill':'blue','font-size':'12','font-weight':'bold'})
    # Marqueurs numérotés
    for idx, (lat, lon) in enumerate(marker_pts):
        color = 'green' if idx == 0 else ('red' if idx == len(marker_pts)-1 else 'blue')
        html = (
            f"<div style='background:{color};border-radius:50%;width:28px;height:28px;"
            f"display:flex;align-items:center;justify-content:center;color:white;font-weight:bold'>"
            f"{idx+1}</div>"
        )
        folium.Marker(
            location=(lat, lon), icon=folium.DivIcon(html=html),
            tooltip=f"Étape {idx+1}: {labels[idx]}", popup=labels[idx]
        ).add_to(m)
    return m

map_obj = create_map(
    route_pts,
    list(zip(df_opt["Latitude"], df_opt["Longitude"])),
    df_opt["Adresse du client"].tolist()
)
st.subheader("Visualisation interactive de la tournée (véhicule)")
st_folium(map_obj, width=800, height=600)

# Export du fichier Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_opt.to_excel(writer, index=False, sheet_name="Repérage")
output.seek(0)

st.download_button(
    "Télécharger le tournée organisée (.xlsx)",
    data=output,
    file_name="tournee_organisee.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
