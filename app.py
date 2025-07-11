import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import io

st.title("Organisateur d'adresses pour repérages")

uploaded_file = st.file_uploader("Chargez votre fichier Excel contenant les adresses", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if "Adresse du client" in df.columns and "CPSTCMN" in df.columns and "LVIL" in df.columns:
        st.success("Fichier reconnu. Les colonnes d'adresses sont valides.")

        df["Adresse complète"] = df["Adresse du client"] + ", " + df["CPSTCMN"].astype(str) + " " + df["LVIL"] + ", France"

        geolocator = Nominatim(user_agent="repérage_web_app")

        latitudes = []
        longitudes = []

        with st.spinner("Géocodage des adresses..."):
            for adresse in df["Adresse complète"]:
                try:
                    location = geolocator.geocode(adresse)
                    if location:
                        latitudes.append(location.latitude)
                        longitudes.append(location.longitude)
                    else:
                        latitudes.append(None)
                        longitudes.append(None)
                except:
                    latitudes.append(None)
                    longitudes.append(None)
                time.sleep(1)

        df["Latitude"] = latitudes
        df["Longitude"] = longitudes

        df_clean = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

        if not df_clean.empty:
            st.success("Géocodage réussi. Organisation en cours...")

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
                    plus_proche_idx = distances.idxmin()
                    parcours.append(reste.loc[plus_proche_idx])
                    reste = reste.drop(plus_proche_idx)

                return pd.DataFrame(parcours)

            df_organise = optimiser_par_proximite(df_clean)

            st.subheader("Adresses organisées selon la proximité :")
            st.dataframe(df_organise[["Adresse du client", "CPSTCMN", "LVIL", "Latitude", "Longitude"]])

            # Générer un fichier Excel en mémoire
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_organise.to_excel(writer, index=False, sheet_name='Repérage')
            output.seek(0)

            st.download_button(
                "Télécharger le fichier organisé", 
                data=output, 
                file_name="reperage_organise.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.error("Aucune adresse valide n'a pu être géocodée.")
    else:
        st.error("Le fichier ne contient pas les colonnes attendues : 'Adresse du client', 'CPSTCMN', 'LVIL'")
