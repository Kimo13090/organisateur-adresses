import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
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

    # Vérification des colonnes
    required_cols = ["Adresse du client", "CPSTCMN", "LVIL"]
    if all(col in df.columns for col in required_cols):
        st.success("Fichier reconnu. Colonnes d'adresses valides.")

        # Création de l'adresse complète
        df["Adresse complète"] = df["Adresse du client"] + ", " + df["CPSTCMN"].astype(str) + " " + df["LVIL"] + ", France"

        # Initialisation du géocodeur
        geolocator = Nominatim(user_agent="repérage_web_app")

        # Fonction de géocodage mise en cache
        @st.cache_data(show_spinner=False)
        def geocode(addr):
            try:
                loc = geolocator.geocode(addr, timeout=10)
                if loc:
                    return loc.latitude, loc.longitude
            except:
                pass
            return None, None

        # Géocodage des adresses
        latitudes, longitudes = [], []
        with st.spinner("Géocodage des adresses..."):
            for adresse in df["Adresse complète"]:
                lat, lon = geocode(adresse)
                latitudes.append(lat)
                longitudes.append(lon)
                time.sleep(0.2)

        df["Latitude"] = latitudes
        df["Longitude"] = longitudes

        # Nettoyage des adresses non géocodées
        df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

        if not df_clean.empty:
            st.success("Géocodage réussi. Organisation en cours...")

            # Optimisation par proximité
            def optimiser_par_proximite(df_input):
                parcours = [df_input.iloc[0]]
                reste = df_input.iloc[1:].copy()
                while not reste.empty:
                    last = parcours[-1]
                    origin = (last["Latitude"], last["Longitude"])
                    distances = reste.apply(
                        lambda row: geodesic(origin, (row["Latitude"], row["Longitude"])).meters,
                        axis=1
                    )
                    idx = distances.idxmin()
                    parcours.append(reste.loc[idx])
                    reste = reste.drop(idx)
                return pd.DataFrame(parcours)

            df_organise = optimiser_par_proximite(df_clean)

            # Affichage du tableau
            st.subheader("Liste organisée")
            st.dataframe(
                df_organise[["Adresse du client", "CPSTCMN", "LVIL", "Latitude", "Longitude"]],
                use_container_width=True
            )

            # Préparation de la carte routière via OSRM
            coords = list(zip(df_organise["Latitude"], df_organise["Longitude"]))
            route_segments = []
            for i in range(len(coords)-1):
                start = coords[i]
                end = coords[i+1]
                url = (
                    f"http://router.project-osrm.org/route/v1/driving/"
                    f"{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
                )
                try:
                    res = requests.get(url, timeout=5).json()
                    seg = res['routes'][0]['geometry']['coordinates']
                    route_segments.extend([(lat, lon) for lon, lat in seg])
                except:
                    route_segments.extend([start, end])

            # Création de la carte interactive
            @st.cache_data(show_spinner=False)
            def create_map(route_pts, marker_pts, addresses):
                m = folium.Map(location=marker_pts[0], zoom_start=12)
                # Tracé de la route avec flèches
                line = folium.PolyLine(route_pts, color="blue", weight=4, opacity=0.7)
                m.add_child(line)
                PolyLineTextPath(
                    line, '▶', repeat=True, offset=10,
                    attributes={'fill': 'blue', 'font-weight': 'bold', 'font-size': '14'}
                ).add_to(m)
                # Marqueurs numérotés avec tooltip et popup
                for idx, (lat, lon) in enumerate(marker_pts):
                    # Couleurs : vert début, rouge fin, bleu intermédiaire
                    color = 'green' if idx == 0 else ('red' if idx == len(marker_pts)-1 else 'blue')
                    icon_html = (
                        f"<div style='background:{color};color:white;border-radius:50%;width:28px;"
                        f"height:28px;line-height:28px;text-align:center;font-weight:bold'>"
                        f"{idx+1}</div>"
                    )
                    folium.Marker(
                        location=(lat, lon),
                        icon=folium.DivIcon(html=icon_html),
                        tooltip=f"Étape {idx+1}: {addresses[idx]}",
                        popup=f"Étape {idx+1}\n{addresses[idx]}"
                    ).add_to(m)
                return m

            map_obj = create_map(route_segments, coords, df_organise["Adresse du client"].tolist())
            st.subheader("Visualisation interactive de la tournée (véhicule)")
            st_folium(map_obj, width=800, height=600)

            # Téléchargement du fichier Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_organise.to_excel(writer, index=False, sheet_name='Repérage')
            output.seek(0)
            st.download_button(
                "Télécharger le fichier organisé en XLSX",
                data=output,
                file_name="reperage_organise.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Aucune adresse valide n'a pu être géocodée.")
    else:
        st.error(f"Colonne(s) manquante(s), nécessaire(s): {required_cols}")
