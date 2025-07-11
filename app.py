import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import io
import folium
from streamlit_folium import st_folium

# Configuration de la page
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")

st.title("Organisateur d'adresses pour repérages")

# Upload du fichier Excel
uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Vérification des colonnes
    if "Adresse du client" in df.columns and "CPSTCMN" in df.columns and "LVIL" in df.columns:
        st.success("Fichier reconnu. Les colonnes d'adresses sont valides.")

        # Création de l'adresse complète
        df["Adresse complète"] = (
            df["Adresse du client"] + ", " +
            df["CPSTCMN"].astype(str) + " " +
            df["LVIL"] + ", France"
        )

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
                time.sleep(0.2)  # pause pour respecter les limites du service

        df["Latitude"] = latitudes
        df["Longitude"] = longitudes

        # Nettoyage des éventuelles adresses non géocodées
        df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

        if not df_clean.empty:
            st.success("Géocodage réussi. Organisation en cours...")

            # Fonction d'optimisation par proximité
            def optimiser_par_proximite(df_input):
                parcours = [df_input.iloc[0]]
                reste = df_input.iloc[1:].copy()
                while not reste.empty:
                    dernier = parcours[-1]
                    coords_dernier = (dernier["Latitude"], dernier["Longitude"])
                    distances = reste.apply(
                        lambda row: geodesic(coords_dernier, (row["Latitude"], row["Longitude"])).meters,
                        axis=1
                    )
                    idx_min = distances.idxmin()
                    parcours.append(reste.loc[idx_min])
                    reste = reste.drop(idx_min)
                return pd.DataFrame(parcours)

            # Application de l'optimisation
            df_organise = optimiser_par_proximite(df_clean)

            # Affichage du tableau organisé
            st.subheader("Adresses organisées selon la proximité :")
            st.dataframe(
                df_organise[["Adresse du client", "CPSTCMN", "LVIL", "Latitude", "Longitude"]],
                use_container_width=True
            )

            # Création de la carte interactive
            coords = list(zip(df_organise["Latitude"], df_organise["Longitude"]))
            m = folium.Map(location=coords[0], zoom_start=12)
            folium.PolyLine(coords, color="blue", weight=4, opacity=0.7).add_to(m)
            for i, (lat, lon) in enumerate(coords):
                folium.Marker(
                    location=(lat, lon),
                    popup=f"{i+1} – {df_organise['Adresse du client'].iloc[i]}",
                    tooltip=str(i+1)
                ).add_to(m)

            st.subheader("Visualisation de la tournée")
            st_folium(m, width=700, height=500)

            # Génération et téléchargement du fichier Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_organise.to_excel(writer, index=False, sheet_name="Repérage")
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
        st.error("Le fichier ne contient pas les colonnes attendues : 'Adresse du client', 'CPSTCMN', 'LVIL'")
