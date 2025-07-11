import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import requests
import folium
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

# Configuration
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Organisateur d'adresses pour repérages")

# Upload Excel
uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# Sidebar: choisir colonnes
st.sidebar.subheader("Configuration des colonnes d'adresse")
cols = df.columns.tolist()
address_col = st.sidebar.selectbox("Colonne adresse", cols)
postal_col = st.sidebar.selectbox("Colonne code postal", cols)
city_col = st.sidebar.selectbox("Colonne ville", cols)

if not address_col or not postal_col or not city_col:
    st.error("Merci de sélectionner les colonnes nécessaires.")
    st.stop()

# Construire adresse complète
df["Adresse complète"] = (
    df[address_col].astype(str) + ", " +
    df[postal_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# Géocodage cache
geolocator = Nominatim(user_agent="repr_api")
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
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(a) for a in df["Adresse complète"]]
# Assigner
lats, lons = zip(*coords)
df["Latitude"], df["Longitude"] = lats, lons

# Filtrer valides
df_clean = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse géocodée avec succès.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées.")

# Déterminer ordre de passage (OSRM Trip ou NN)
with st.spinner("Calcul de l'ordre de passage..."):
    try:
        waypoints = ";".join(f"{lon},{lat}" for lat, lon in zip(df_clean["Latitude"], df_clean["Longitude"]))
        trip_url = f"http://router.project-osrm.org/trip/v1/driving/{waypoints}?source=first&roundtrip=false&overview=full&geometries=geojson"
        r = requests.get(trip_url, timeout=15)
        r.raise_for_status()
        trip = r.json()["trips"][0]
        order = trip.get("waypoint_order", list(range(len(df_clean))))
    except Exception:
        # Fallback nearest neighbor
        def nn(df_nn):
            seq = [df_nn.iloc[0]]
            rem = df_nn.iloc[1:].copy().reset_index(drop=True)
            while not rem.empty:
                last = seq[-1]
                o = (last["Latitude"], last["Longitude"])
                dists = rem.apply(lambda r: geodesic(o,(r["Latitude"],r["Longitude"])).meters, axis=1)
                i = dists.idxmin()
                seq.append(rem.loc[i])
                rem = rem.drop(i).reset_index(drop=True)
            return pd.DataFrame(seq)
        df_ordered_nn = nn(df_clean)
        order = df_ordered_nn.index.tolist()

# DataFrame ordonné
df_opt = df_clean.iloc[order].reset_index(drop=True)

# Affichage table organisée
st.subheader("Adresses organisées")
st.dataframe(df_opt[[address_col,postal_col,city_col,"Latitude","Longitude"]], use_container_width=True)

# Calcul et tracé de la route réelle via OSRM segment par segment
@st.cache_data
def build_route(pts):
    route = []
    for i in range(len(pts)-1):
        start = pts[i]
        end = pts[i+1]
        url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            seg = res.json()["routes"][0]["geometry"]["coordinates"]
            route.extend([(lat,lon) for lon,lat in seg])
        except:
            # Trajet direct si échec
            route.append(start)
            route.append(end)
    return route

coords_list = list(zip(df_opt["Latitude"], df_opt["Longitude"]))
route_pts = build_route(coords_list)

# Carte interactive
m = folium.Map(location=coords_list[0], zoom_start=12)
line = folium.PolyLine(route_pts, color="blue", weight=4, opacity=0.7)
m.add_child(line)
PolyLineTextPath(line, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})
for idx,(lat,lon) in enumerate(coords_list):
    color = 'green' if idx==0 else ('red' if idx==len(coords_list)-1 else 'blue')
    html = f"<div style='background:{color};color:white;border-radius:50%;width:24px;height:24px;"
            f"display:flex;align-items:center;justify-content:center;font-weight:bold'>{idx+1}</div>"
    folium.Marker(location=(lat,lon),icon=folium.DivIcon(html=html),tooltip=df_opt[address_col].iloc[idx]).add_to(m)
st.subheader("Itinéraire interactif (voiture)")
st_folium(m,width=800,height=600)

# Export Excel
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df_opt.to_excel(writer, index=False, sheet_name="Repérage")
out.seek(0)
st.download_button("Télécharger (.xlsx)", data=out, file_name="tournee_organisee.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
