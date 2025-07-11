import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import time
import numpy as np
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Organisateur de Tournées",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🚚 Organisateur de Tournées Optimisées")
st.markdown("---")

# Fonction pour calculer la distance totale d'une tournée
def calculate_total_distance(df_ordered):
    """Calcule la distance totale d'une tournée"""
    total_distance = 0
    for i in range(len(df_ordered) - 1):
        coord1 = (df_ordered.iloc[i]['lat'], df_ordered.iloc[i]['lon'])
        coord2 = (df_ordered.iloc[i + 1]['lat'], df_ordered.iloc[i + 1]['lon'])
        total_distance += geodesic(coord1, coord2).kilometers
    return total_distance

# Fonction de géocodage avec retry
@st.cache_data
def geocode_address(address, max_retries=3):
    """Géocode une adresse avec retry en cas d'échec"""
    geolocator = Nominatim(user_agent="streamlit_route_organizer_v2")
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            else:
                return (None, None)
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Erreur géocodage pour '{address}': {e}")
                return (None, None)
            time.sleep(1)  # Pause avant retry
    
    return (None, None)

# Algorithme du plus proche voisin amélioré
def optimize_route_nearest_neighbor(df_points, start_method='central'):
    """
    Optimise la route avec l'algorithme du plus proche voisin
    start_method: 'central', 'first', ou 'custom'
    """
    if len(df_points) <= 1:
        return df_points.index.tolist()
    
    # Déterminer le point de départ
    if start_method == 'central':
        # Point le plus central (minimise la somme des distances)
        distances_sum = []
        for i in df_points.index:
            total_dist = 0
            for j in df_points.index:
                if i != j:
                    coord1 = (df_points.at[i, 'lat'], df_points.at[i, 'lon'])
                    coord2 = (df_points.at[j, 'lat'], df_points.at[j, 'lon'])
                    total_dist += geodesic(coord1, coord2).kilometers
            distances_sum.append(total_dist)
        
        start_idx = df_points.index[np.argmin(distances_sum)]
    else:
        # Premier point du dataset
        start_idx = df_points.index[0]
    
    # Algorithme du plus proche voisin
    route_order = [start_idx]
    remaining_points = set(df_points.index) - {start_idx}
    
    current_point = start_idx
    
    while remaining_points:
        current_coord = (df_points.at[current_point, 'lat'], df_points.at[current_point, 'lon'])
        
        # Trouver le point le plus proche
        min_distance = float('inf')
        next_point = None
        
        for point in remaining_points:
            point_coord = (df_points.at[point, 'lat'], df_points.at[point, 'lon'])
            distance = geodesic(current_coord, point_coord).kilometers
            
            if distance < min_distance:
                min_distance = distance
                next_point = point
        
        route_order.append(next_point)
        remaining_points.remove(next_point)
        current_point = next_point
    
    return route_order

# Interface utilisateur
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Chargement des données")
    uploaded_file = st.file_uploader(
        "Chargez votre fichier Excel (.xlsx)",
        type=["xlsx"],
        help="Le fichier doit contenir au minimum une colonne adresse, code postal et ville"
    )

with col2:
    st.header("⚙️ Paramètres")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Fichier chargé: {len(df)} lignes")
        except Exception as e:
            st.error(f"❌ Erreur lecture fichier: {e}")
            st.stop()
    else:
        st.info("En attente du fichier Excel...")
        st.stop()

# Affichage des données brutes
if not df.empty:
    st.markdown("---")
    st.subheader("📊 Aperçu des données")
    st.dataframe(df.head(), use_container_width=True)

# Configuration des colonnes
st.sidebar.header("🗂️ Configuration des colonnes")
columns = df.columns.tolist()

address_col = st.sidebar.selectbox("Colonne Adresse", columns, key="addr")
postal_col = st.sidebar.selectbox("Colonne Code Postal", columns, key="postal")
city_col = st.sidebar.selectbox("Colonne Ville", columns, key="city")

# Validation des colonnes
if not all([address_col, postal_col, city_col]):
    st.error("⚠️ Veuillez sélectionner toutes les colonnes requises")
    st.stop()

# Paramètres d'optimisation
st.sidebar.header("🎯 Paramètres d'optimisation")
start_method = st.sidebar.radio(
    "Point de départ",
    ["central", "first"],
    format_func=lambda x: "Point central" if x == "central" else "Premier point"
)

filter_outliers = st.sidebar.checkbox("Filtrer les points aberrants", value=True)
if filter_outliers:
    outlier_threshold = st.sidebar.slider("Seuil de filtrage (écarts-types)", 1.0, 3.0, 1.5)

# Bouton de traitement
if st.sidebar.button("🚀 Organiser la tournée", type="primary"):
    
    # Construction de l'adresse complète
    df['adresse_complete'] = (
        df[address_col].astype(str) + ", " + 
        df[postal_col].astype(str) + " " + 
        df[city_col].astype(str) + ", France"
    )
    
    # Géocodage
    st.markdown("---")
    st.subheader("📍 Géocodage des adresses")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    coordinates = []
    total_addresses = len(df)
    
    for i, address in enumerate(df['adresse_complete']):
        progress = (i + 1) / total_addresses
        progress_bar.progress(progress)
        status_text.text(f"Géocodage en cours... {i+1}/{total_addresses}")
        
        lat, lon = geocode_address(address)
        coordinates.append((lat, lon))
    
    # Ajout des coordonnées au DataFrame
    df[['lat', 'lon']] = pd.DataFrame(coordinates)
    
    # Filtrage des adresses non géocodées
    df_geocoded = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
    failed_geocoding = len(df) - len(df_geocoded)
    
    if failed_geocoding > 0:
        st.warning(f"⚠️ {failed_geocoding} adresse(s) n'ont pas pu être géocodées")
    
    if df_geocoded.empty:
        st.error("❌ Aucune adresse n'a pu être géocodée")
        st.stop()
    
    st.success(f"✅ {len(df_geocoded)} adresses géocodées avec succès")
    
    # Filtrage des points aberrants
    if filter_outliers and len(df_geocoded) > 2:
        centroid = (df_geocoded['lat'].mean(), df_geocoded['lon'].mean())
        
        df_geocoded['distance_centroid'] = df_geocoded.apply(
            lambda row: geodesic((row['lat'], row['lon']), centroid).kilometers,
            axis=1
        )
        
        mean_distance = df_geocoded['distance_centroid'].mean()
        std_distance = df_geocoded['distance_centroid'].std()
        threshold_distance = mean_distance + outlier_threshold * std_distance
        
        df_main = df_geocoded[df_geocoded['distance_centroid'] <= threshold_distance].reset_index(drop=True)
        df_outliers = df_geocoded[df_geocoded['distance_centroid'] > threshold_distance].reset_index(drop=True)
        
        if not df_outliers.empty:
            st.warning(f"⚠️ {len(df_outliers)} point(s) aberrant(s) détecté(s)")
            with st.expander("Voir les points aberrants"):
                st.dataframe(df_outliers[[address_col, postal_col, city_col, 'distance_centroid']])
    else:
        df_main = df_geocoded.copy()
        df_outliers = pd.DataFrame()
    
    if df_main.empty:
        st.error("❌ Aucune adresse valide après filtrage")
        st.stop()
    
    # Optimisation de la tournée
    st.markdown("---")
    st.subheader("🎯 Optimisation de la tournée")
    
    with st.spinner("Calcul de l'itinéraire optimal..."):
        optimal_order = optimize_route_nearest_neighbor(df_main, start_method)
        df_optimized = df_main.loc[optimal_order].reset_index(drop=True)
    
    # Calcul des statistiques
    total_distance = calculate_total_distance(df_optimized)
    
    # Affichage des résultats
    st.markdown("---")
    st.subheader("📋 Résultats de l'optimisation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏠 Nombre d'adresses", len(df_optimized))
    with col2:
        st.metric("📏 Distance totale", f"{total_distance:.1f} km")
    with col3:
        st.metric("⏱️ Temps estimé", f"{total_distance * 2:.0f} min")
    
    # Tableau des résultats
    st.subheader("📊 Tournée optimisée")
    
    # Ajout d'un numéro d'ordre
    df_display = df_optimized.copy()
    df_display.insert(0, 'Ordre', range(1, len(df_display) + 1))
    
    # Colonnes à afficher
    display_columns = ['Ordre', address_col, postal_col, city_col, 'lat', 'lon']
    
    st.dataframe(
        df_display[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Export Excel
    st.markdown("---")
    st.subheader("💾 Export des résultats")
    
    # Préparation des données pour l'export
    export_data = df_optimized.copy()
    export_data.insert(0, 'Ordre_visite', range(1, len(export_data) + 1))
    
    # Création du fichier Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille principale
        export_data.to_excel(writer, index=False, sheet_name='Tournée_optimisée')
        
        # Feuille avec points aberrants si applicable
        if not df_outliers.empty:
            df_outliers.to_excel(writer, index=False, sheet_name='Points_aberrants')
    
    output.seek(0)
    
    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tournee_optimisee_{timestamp}.xlsx"
    
    st.download_button(
        label="📥 Télécharger la tournée optimisée",
        data=output,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type="primary"
    )
    
    # Informations supplémentaires
    with st.expander("ℹ️ Informations détaillées"):
        st.write(f"**Méthode d'optimisation:** Plus proche voisin")
        st.write(f"**Point de départ:** {start_method}")
        st.write(f"**Filtrage aberrants:** {'Activé' if filter_outliers else 'Désactivé'}")
        if filter_outliers:
            st.write(f"**Seuil de filtrage:** {outlier_threshold} écart(s)-type(s)")
        st.write(f"**Adresses traitées:** {len(df_optimized)}")
        st.write(f"**Adresses échouées:** {failed_geocoding}")
        if not df_outliers.empty:
            st.write(f"**Points aberrants:** {len(df_outliers)}")

# Footer
st.markdown("---")
st.markdown("🔧 **Organisateur de Tournées** - Optimisez vos itinéraires facilement!")
