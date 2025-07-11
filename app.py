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
uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    required_cols = ["Adresse du client", "CPSTCMN", "LVIL"]
    if all(col in df.columns for col in required_cols):
        st.success("Fichier reconnu. Colonnes valides.")
        # Construction de l'adresse complète
        df["Adresse complète"] = (
            df["Adresse du client"] + ", " +
            df["CPSTCMN"].astype(str) + " " +
            df["LVIL"] + ", France"
        )
        # Géocodeur avec cache
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
        with st.spinner("Géocodage des adresses..."):
            coords = [geocode(addr) for addr in df["Adresse complète"]]
        lats, lons = zip(*coords)
        df["Latitude"] = lats
        df["Longitude"] = lons
        # Nettoyage
        df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        if df_clean.empty:
            st.error("Aucune adresse géocodée.")
        else:
            st.success("Géocodage OK, optimisation en cours...")
            # Nearest neighbor optimization
            def optimiser(df_in):
                seq = [df_in.iloc[0]]
                reste = df_in.iloc[1:].copy()
                while not reste.empty:
                    last = seq[-1]
                    origin = (last["Latitude"], last["Longitude"])
                    idx = reste.apply(
                        lambda row: geodesic(origin, (row["Latitude"], row["Longitude"])).meters,
                        axis=1
                    ).idxmin()
                    seq.append(reste.loc[idx])
                    reste = reste.drop(idx)
                return pd.DataFrame(seq)
            df_opt = optimiser(df_clean)
            st.subheader("Liste organisée")
            st.dataframe(
                df_opt[["Adresse du client", "CPSTCMN", "LVIL", "Latitude", "Longitude"]],
                use_container_width=True
            )
            # Appel OSRM unique
            waypoints = ";".join(
                f"{lon},{lat}" for lat, lon in zip(df_opt["Latitude"], df_opt["Longitude"])
            )
            url = (
                f"http://router.project-osrm.org/route/v1/driving/{waypoints}"
                f"?overview=simplified&geometries=geojson"
            )
            try:
                resp = requests.get(url, timeout=10).json()
                route_geo = resp["routes"][0]["geometry"]["coordinates"]
                route_pts = [(lat, lon) for lon, lat in route_geo]
            except:
                route_pts = list(zip(df_opt["Latitude"], df_opt["Longitude"]))
            # Création de la carte
            @st.cache_data
            def make_map(route_pts, markers, labels):
                m = folium.Map(location=markers[0], zoom_start=12)
                line = folium.PolyLine(route_pts, color="blue", weight=4, opacity=0.7)
                m.add_child(line)
                PolyLineTextPath(
                    line, '▶', repeat=True, offset=12,
                    attributes={"fill":"blue","font-weight":"bold","font-size":"14"}
                ).add_to(m)
                for i, (lat, lon) in enumerate(markers):
                    color = "green" if i == 0 else ("red" if i == len(markers)-1 else "blue")
                    icon = folium.DivIcon(html=
                        f"<div style='background:{color};border-radius:50%;width:28px;height:28px;"
                        f"display:flex;align-items:center;justify-content:center;color:white;font-weight:bold'>"
                        f"{i+1}</div>"
                    )
                    folium.Marker(
                        location=(lat, lon),
                        icon=icon,
                        tooltip=labels[i],
                        popup=labels[i]
                    ).add_to(m)
                return m
            map_obj = make_map(
                route_pts,
                list(zip(df_opt["Latitude"], df_opt["Longitude"])),
                df_opt["Adresse du client"].tolist()
            )
            st.subheader("Carte interactive de la tournée (véhicule)")
            st_folium(map_obj, width=800, height=600)
            # Téléchargement Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_opt.to_excel(writer, index=False, sheet_name="Repérage")
            output.seek(0)
            st.download_button(
                "Télécharger en XLSX",
                data=output,
                file_name="repérage_organisé.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        missing = list(set(required_cols) - set(df.columns))
        st.error(f"Colonnes manquantes: {', '.join(missing)}")
