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

# Upload du fichier Excel
uploaded_file = st.file_uploader(
    "Chargez votre fichier Excel contenant les adresses", type=["xlsx"]
)
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

# Construire l'adresse complète
df["Adresse complète"] = (
    df[address_col].astype(str) + ", " +
    df[postal_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# Géocodeur OSM avec cache
g = Nominatim(user_agent="repérage_web_app")
@st.cache_data
 def geocode(addr):
    try:
        loc = g.geocode(addr, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

# Géocodage
with st.spinner("Géocodage des adresses..."):
    coords = [geocode(a) for a in df["Adresse complète"]]

# Assigner lat/lon
lats, lons = zip(*coords)
df["Latitude"], df["Longitude"] = lats, lons

# Filtrer adresses valides
df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
if df_clean.empty:
    st.error("Aucune adresse n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df_clean)} adresses géocodées avec succès.")

# Titre de la tournée (départ = 1ère ligne)
st.sidebar.subheader("Options d'itinéraire")
start_index = st.sidebar.selectbox(
    "Point de départ (index)", list(range(len(df_clean))), index=0
)

# Optimisation via OSRM Trip
with st.spinner("Optimisation de la tournée (OSRM Trip)..."):
    wpts = ";".join(
        f"{lon},{lat}" for lat, lon in zip(df_clean["Latitude"], df_clean["Longitude"])
    )
    url = (
        f"http://router.project-osrm.org/trip/v1/driving/{wpts}"
        f"?source=first&roundtrip=false&overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        trip = data.get("trips", [])[0]
        order = trip.get("waypoint_order", list(range(len(df_clean))))
        coords_geo = trip["geometry"]["coordinates"]
        route_pts = [(lat, lon) for lon, lat in coords_geo]
        df_opt = df_clean.iloc[order].reset_index(drop=True)
    except:
        # Fallback NN
        def nn(seq_df, start=0):
            seq = [seq_df.iloc[start]]
            rem = seq_df.drop(start).reset_index(drop=True)
            while not rem.empty:
                last = seq[-1]
                o = (last["Latitude"], last["Longitude"])
                dists = rem.apply(lambda r: geodesic(o, (r["Latitude"], r["Longitude"])).meters, axis=1)
                mi = dists.idxmin()
                seq.append(rem.loc[mi])
                rem = rem.drop(mi).reset_index(drop=True)
            return pd.DataFrame(seq)
        df_opt = nn(df_clean, start=start_index)
        route_pts = list(zip(df_opt["Latitude"], df_opt["Longitude"]))

# Afficher le tableau optimisé
st.subheader("Adresses organisées pour la tournée")
st.dataframe(df_opt[[address_col, postal_col, city_col, "Latitude", "Longitude"]], use_container_width=True)

# Création de la carte
@st.cache_data
 def make_map(route, markers, labels):
    m = folium.Map(location=markers[0], zoom_start=12)
    pl = folium.PolyLine(route, color="blue", weight=4, opacity=0.7)
    m.add_child(pl)
    PolyLineTextPath(pl, '▶', repeat=True, offset=10, attributes={'fill':'blue','font-size':'12'})
    for i, (lat, lon) in enumerate(markers):
        col = 'green' if i==0 else ('red' if i==len(markers)-1 else 'blue')
        html = f"<div style='background:{col};color:white;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:bold'>{i+1}</div>"
        folium.Marker(location=(lat,lon), icon=folium.DivIcon(html=html), tooltip=f"Étape {i+1}: {labels[i]}").add_to(m)
    return m

map_obj = make_map(
    route_pts,
    list(zip(df_opt["Latitude"], df_opt["Longitude"])),
    df_opt[address_col].tolist()
)
st.subheader("Visualisation interactive de la tournée (véhicule)")
st_folium(map_obj, width=800, height=600)

# Export Excel
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
