import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import io
import time
import numpy as np
from datetime import datetime
import math

# Configuration de la page avec thème personnalisé
st.set_page_config(
    page_title="Optimisateur de Tournées Logistiques",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialisation du session state
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'df_optimized' not in st.session_state:
    st.session_state.df_optimized = None
if 'df_out_sector' not in st.session_state:
    st.session_state.df_out_sector = None
if 'df_failed' not in st.session_state:
    st.session_state.df_failed = None
if 'total_distance' not in st.session_state:
    st.session_state.total_distance = 0
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None

# CSS personnalisé pour améliorer le design
st.markdown("""
<style>
    /* Thème général */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Titre principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Cartes métriques */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Boutons personnalisés */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Conteneur de résultats fixe */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Section de téléchargement mise en avant */
    .download-section {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        background: #2d3748;
        color: white;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 3rem;
    }
    
    /* Animation de succès */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .success-message {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# En-tête personnalisé
st.markdown("""
<div class="main-header">
    <h1>🚛 Optimisateur de Tournées Logistiques</h1>
    <p>Solution intelligente pour l'optimisation automatique de vos itinéraires de livraison</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def geocode_address(address, max_retries=3):
    """Géocode une adresse avec retry et gestion d'erreurs"""
    geolocator = Nominatim(user_agent="logistics_optimizer_v2")
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=15)
            if location:
                return (location.latitude, location.longitude, True)
            else:
                return (None, None, False)
        except Exception as e:
            if attempt == max_retries - 1:
                return (None, None, False)
            time.sleep(2)
    
    return (None, None, False)

def find_outliers_using_mad(df_points):
    """Détecte automatiquement les adresses hors secteur"""
    if len(df_points) < 5:
        return df_points.index.tolist(), []
    
    center_lat = df_points['lat'].median()
    center_lon = df_points['lon'].median()
    center = (center_lat, center_lon)
    
    distances = []
    for idx, row in df_points.iterrows():
        point = (row['lat'], row['lon'])
        dist = geodesic(center, point).kilometers
        distances.append(dist)
    
    df_points['distance_from_center'] = distances
    
    median_distance = np.median(distances)
    mad = np.median(np.abs(distances - median_distance))
    threshold = median_distance + 2.5 * mad
    threshold = max(threshold, 10.0)
    
    in_sector = []
    out_sector = []
    
    for idx, dist in zip(df_points.index, distances):
        if dist <= threshold:
            in_sector.append(idx)
        else:
            out_sector.append(idx)
    
    return in_sector, out_sector

def create_folium_map(df_route, df_out_sector=None):
    """Crée une carte Folium interactive avec l'itinéraire"""
    center_lat = df_route['lat'].mean()
    center_lon = df_route['lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    for i, row in df_route.iterrows():
        if i == 0:
            color = 'green'
            icon = 'play'
            prefix = 'fa'
        elif i == len(df_route) - 1:
            color = 'red'
            icon = 'stop'
            prefix = 'fa'
        else:
            color = 'blue'
            icon = 'location-dot'
            prefix = 'fa'
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"""
            <b>Ordre: {i+1}</b><br>
            {row.get('adresse_complete', 'Adresse')}<br>
            <i>Distance depuis précédent: {row.get('distance_etape', 'N/A'):.1f} km</i>
            """,
            tooltip=f"Stop {i+1}",
            icon=folium.Icon(color=color, icon=icon, prefix=prefix)
        ).add_to(m)
    
    route_coords = [[row['lat'], row['lon']] for _, row in df_route.iterrows()]
    folium.PolyLine(
        route_coords,
        color='purple',
        weight=3,
        opacity=0.8,
        smooth_factor=2
    ).add_to(m)
    
    if df_out_sector is not None and len(df_out_sector) > 0:
        for _, row in df_out_sector.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"""
                <b>HORS SECTEUR</b><br>
                {row.get('adresse_complete', 'Adresse')}<br>
                <i>Distance du centre: {row.get('distance_from_center', 'N/A'):.1f} km</i>
                """,
                tooltip="Hors secteur",
                icon=folium.Icon(color='orange', icon='exclamation-triangle', prefix='fa')
            ).add_to(m)
    
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto;
                background-color: white; z-index: 1000; 
                border: 2px solid grey; border-radius: 5px;
                padding: 10px; font-size: 14px;">
        <p style="margin: 0; font-weight: bold;">Légende</p>
        <p style="margin: 5px 0;"><i class="fa fa-play" style="color:green;"></i> Départ</p>
        <p style="margin: 5px 0;"><i class="fa fa-location-dot" style="color:blue;"></i> Livraison</p>
        <p style="margin: 5px 0;"><i class="fa fa-stop" style="color:red;"></i> Arrivée</p>
        <p style="margin: 5px 0;"><i class="fa fa-exclamation-triangle" style="color:orange;"></i> Hors secteur</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def optimize_route_nearest_neighbor(df_points):
    """Optimisation par plus proche voisin améliorée"""
    if len(df_points) <= 1:
        return df_points.index.tolist()
    
    center_lat = df_points['lat'].mean()
    center_lon = df_points['lon'].mean()
    
    min_dist = float('inf')
    start_idx = df_points.index[0]
    
    for idx, row in df_points.iterrows():
        dist = geodesic((center_lat, center_lon), (row['lat'], row['lon'])).kilometers
        if dist < min_dist:
            min_dist = dist
            start_idx = idx
    
    route = [start_idx]
    unvisited = set(df_points.index) - {start_idx}
    
    while unvisited:
        current_idx = route[-1]
        current_coords = (df_points.at[current_idx, 'lat'], df_points.at[current_idx, 'lon'])
        
        nearest_idx = None
        nearest_dist = float('inf')
        
        for idx in unvisited:
            point_coords = (df_points.at[idx, 'lat'], df_points.at[idx, 'lon'])
            dist = geodesic(current_coords, point_coords).kilometers
            
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        
        route.append(nearest_idx)
        unvisited.remove(nearest_idx)
    
    return route

def calculate_route_distances(df_route):
    """Calcule les distances entre chaque étape"""
    distances = []
    cumulative_dist = 0
    
    for i in range(len(df_route)):
        if i == 0:
            distances.append(0)
        else:
            coord1 = (df_route.iloc[i-1]['lat'], df_route.iloc[i-1]['lon'])
            coord2 = (df_route.iloc[i]['lat'], df_route.iloc[i]['lon'])
            dist = geodesic(coord1, coord2).kilometers
            distances.append(dist)
            cumulative_dist += dist
    
    df_route['distance_etape'] = distances
    df_route['distance_cumulee'] = df_route['distance_etape'].cumsum()
    
    return df_route, cumulative_dist

def detect_columns_smart(df):
    """Détection intelligente des colonnes"""
    columns = df.columns.tolist()
    
    address_keywords = ['adresse', 'address', 'rue', 'street', 'voie', 'client', 'nom', 'lieu']
    postal_keywords = ['postal', 'cp', 'code', 'zip', 'postcode']
    city_keywords = ['ville', 'city', 'commune', 'localite', 'locality']
    
    address_col = None
    postal_col = None
    city_col = None
    
    for col in columns:
        col_lower = col.lower()
        
        if not address_col:
            for keyword in address_keywords:
                if keyword in col_lower:
                    address_col = col
                    break
        
        if not postal_col:
            for keyword in postal_keywords:
                if keyword in col_lower:
                    postal_col = col
                    break
        
        if not city_col:
            for keyword in city_keywords:
                if keyword in col_lower:
                    city_col = col
                    break
    
    return address_col, postal_col, city_col

def generate_excel_file(df_optimized, df_out_sector, df_failed, total_distance, estimated_time):
    """Génère le fichier Excel avec toutes les données"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Préparation de la feuille principale
        export_main = df_optimized.copy()
        export_main.insert(0, 'Ordre', range(1, len(export_main) + 1))
        export_main['Secteur'] = 'Principal'
        export_main['Distance étape (km)'] = export_main['distance_etape'].round(1)
        export_main['Heure estimée'] = pd.to_datetime('08:00:00') + pd.to_timedelta(
            export_main.index * 5 + export_main['distance_cumulee'] * 3, unit='minutes'
        )
        export_main['Heure estimée'] = export_main['Heure estimée'].dt.strftime('%H:%M')
        
        # Colonnes à garder
        cols_to_keep = ['Ordre', 'Secteur'] + [col for col in export_main.columns if col in ['adresse', 'address', 'rue', 'street', 'voie', 'client', 'nom', 'lieu']]
        cols_to_keep += [col for col in export_main.columns if any(keyword in col.lower() for keyword in ['postal', 'cp', 'code', 'zip'])]
        cols_to_keep += [col for col in export_main.columns if any(keyword in col.lower() for keyword in ['ville', 'city', 'commune'])]
        cols_to_keep += ['Distance étape (km)', 'Heure estimée', 'lat', 'lon']
        
        # Éliminer les doublons
        cols_to_keep = list(dict.fromkeys([col for col in cols_to_keep if col in export_main.columns]))
        
        export_main[cols_to_keep].to_excel(writer, index=False, sheet_name='Itinéraire_Optimisé')
        
        # Feuille des adresses hors secteur
        if len(df_out_sector) > 0:
            export_out = df_out_sector.copy()
            export_out['Secteur'] = 'HORS SECTEUR'
            export_out['Distance du centre (km)'] = export_out['distance_from_center'].round(1)
            export_out.to_excel(writer, index=False, sheet_name='Hors_Secteur')
        
        # Feuille des échecs
        if len(df_failed) > 0:
            df_failed.to_excel(writer, index=False, sheet_name='Échecs_Géolocalisation')
        
        # Feuille de synthèse
        summary_data = {
            'Métrique': ['Total adresses', 'Adresses géolocalisées', 'Dans le secteur', 'Hors secteur', 
                        'Distance totale (km)', 'Temps estimé (min)', 'Heure de départ', 'Heure d\'arrivée'],
            'Valeur': [len(df_optimized) + len(df_out_sector) + len(df_failed), 
                      len(df_optimized) + len(df_out_sector), 
                      len(df_optimized), 
                      len(df_out_sector),
                      round(total_distance, 1), 
                      int(estimated_time), 
                      '08:00',
                      (pd.to_datetime('08:00:00') + pd.to_timedelta(estimated_time, unit='minutes')).strftime('%H:%M')]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Synthèse')
    
    output.seek(0)
    return output

# Interface principale - Upload de fichier
st.markdown("### 📁 Importez votre fichier Excel")

uploaded_file = st.file_uploader(
    "Glissez-déposez votre fichier contenant les adresses clients",
    type=["xlsx", "xls"],
    help="Format requis : colonnes Adresse, Code Postal et Ville",
    key="file_uploader"
)

# Si des résultats existent déjà, les afficher
if st.session_state.processing_done and st.session_state.df_optimized is not None:
    st.markdown("---")
    st.success("✅ **Itinéraire déjà calculé !** Descendez pour voir les résultats ou importez un nouveau fichier.")
    
    # Section de téléchargement en haut pour éviter le scroll
    st.markdown("""
    <div class="download-section">
        <h3 style="margin: 0 0 1rem 0;">📥 Téléchargement disponible</h3>
        <p style="margin: 0;">Votre fichier Excel optimisé est prêt !</p>
    </div>
    """, unsafe_allow_html=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"tournee_optimisee_{timestamp}.xlsx"
    
    st.download_button(
        label="📥 **Télécharger le fichier Excel optimisé**",
        data=st.session_state.excel_data,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type="primary",
        use_container_width=True,
        key="download_top"
    )
    
    # Affichage des résultats sauvegardés
    st.markdown("### 📊 Résultats de l'optimisation")
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Livraisons", len(st.session_state.df_optimized))
    with col2:
        st.metric("📏 Distance totale", f"{st.session_state.total_distance:.1f} km")
    with col3:
        estimated_time = st.session_state.total_distance * 3 + len(st.session_state.df_optimized) * 5
        st.metric("⏱️ Temps estimé", f"{int(estimated_time//60)}h {int(estimated_time%60)}min")
    with col4:
        st.metric("⚠️ Hors secteur", len(st.session_state.df_out_sector))
    
    # Carte
    st.markdown("### 🗺️ Visualisation de la tournée")
    if st.session_state.map_data:
        st_folium(st.session_state.map_data, height=500, use_container_width=True, key="map_saved")
    
    # Tableau de l'itinéraire
    st.markdown("### 📋 Détail de l'itinéraire")
    df_display = st.session_state.df_optimized.copy()
    df_display.insert(0, 'Ordre', range(1, len(df_display) + 1))
    
    # Sélection des colonnes à afficher
    cols_to_display = ['Ordre']
    for col in df_display.columns:
        if any(keyword in col.lower() for keyword in ['adresse', 'address', 'client', 'postal', 'cp', 'ville', 'city']):
            if col not in cols_to_display:
                cols_to_display.append(col)
    
    if 'distance_etape' in df_display.columns:
        df_display['Distance étape'] = df_display['distance_etape'].round(1).astype(str) + ' km'
        cols_to_display.append('Distance étape')
    
    st.dataframe(df_display[cols_to_display], use_container_width=True, hide_index=True)
    
    st.markdown("---")

# Traitement du nouveau fichier
if uploaded_file:
    # Réinitialiser si un nouveau fichier est uploadé
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.processing_done = False
        st.session_state.last_uploaded_file = uploaded_file.name
    
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Fichier importé : **{len(df)} adresses** détectées")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        st.stop()
    
    # Aperçu
    with st.expander("📊 Aperçu des données", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Détection des colonnes
    address_col, postal_col, city_col = detect_columns_smart(df)
    
    if not all([address_col, postal_col, city_col]):
        st.warning("⚠️ Sélection manuelle des colonnes requise")
        
        col1, col2, col3 = st.columns(3)
        columns = df.columns.tolist()
        
        with col1:
            address_col = st.selectbox("📍 Adresse", columns, key="addr_col")
        with col2:
            postal_col = st.selectbox("📮 Code Postal", columns, key="postal_col")
        with col3:
            city_col = st.selectbox("🏙️ Ville", columns, key="city_col")
    else:
        st.info(f"✅ Colonnes détectées : **{address_col}**, **{postal_col}**, **{city_col}**")
    
    # Bouton de traitement
    if st.button("🚀 **Générer l'itinéraire optimisé**", type="primary", use_container_width=True, key="process_btn"):
        
        # Nettoyage
        df_clean = df.dropna(subset=[address_col, postal_col, city_col]).copy()
        df_clean[postal_col] = df_clean[postal_col].astype(str).str.extract('(\d+)')[0]
        df_clean = df_clean[df_clean[address_col].astype(str).str.strip() != '']
        df_clean = df_clean[df_clean[city_col].astype(str).str.strip() != '']
        
        # Construction des adresses
        df_clean['adresse_complete'] = (
            df_clean[address_col].astype(str) + ", " + 
            df_clean[postal_col].astype(str) + " " + 
            df_clean[city_col].astype(str) + ", France"
        )
        
        # Géocodage
        st.markdown("### 🌍 Géolocalisation en cours...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        coordinates = []
        geocoding_success = []
        total = min(len(df_clean), 100)  # Limite gratuite
        
        if len(df_clean) > 100:
            st.warning("⚠️ Limitation à 100 adresses (API gratuite)")
            df_clean = df_clean.head(100)
        
        for i, address in enumerate(df_clean['adresse_complete']):
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Traitement : {i+1}/{total}")
            
            lat, lon, success = geocode_address(address)
            coordinates.append((lat, lon))
            geocoding_success.append(success)
            
            time.sleep(1.1)
        
        df_clean[['lat', 'lon']] = pd.DataFrame(coordinates)
        df_clean['geocoding_success'] = geocoding_success
        
        # Séparation des résultats
        df_geocoded = df_clean[df_clean['geocoding_success'] == True].dropna(subset=['lat', 'lon']).copy()
        df_failed = df_clean[df_clean['geocoding_success'] == False].copy()
        
        if len(df_geocoded) < 2:
            st.error("❌ Pas assez d'adresses géolocalisées")
            st.stop()
        
        # Détection des hors secteur
        with st.spinner("Analyse du secteur..."):
            in_sector_idx, out_sector_idx = find_outliers_using_mad(df_geocoded)
        
        df_sector = df_geocoded.loc[in_sector_idx].copy()
        df_out_sector = df_geocoded.loc[out_sector_idx].copy()
        
        # Optimisation
        with st.spinner("Optimisation de l'itinéraire..."):
            optimal_order = optimize_route_nearest_neighbor(df_sector)
            df_optimized = df_sector.loc[optimal_order].reset_index(drop=True)
            df_optimized, total_distance = calculate_route_distances(df_optimized)
        
        # Génération de la carte
        folium_map = create_folium_map(df_optimized, df_out_sector)
        
        # Génération du fichier Excel
        estimated_time = total_distance * 3 + len(df_optimized) * 5
        excel_data = generate_excel_file(df_optimized, df_out_sector, df_failed, total_distance, estimated_time)
        
        # Sauvegarde dans session state
        st.session_state.processing_done = True
        st.session_state.df_optimized = df_optimized
        st.session_state.df_out_sector = df_out_sector
        st.session_state.df_failed = df_failed
        st.session_state.total_distance = total_distance
        st.session_state.excel_data = excel_data
        st.session_state.map_data = folium_map
        
        # Forcer le rechargement pour afficher les résultats
        st.rerun()

# Message d'attente si pas de fichier
if not uploaded_file and not st.session_state.processing_done:
    st.markdown("""
    <div class="results-container">
        <h4>📋 Format attendu du fichier Excel :</h4>
        <ul>
            <li><b>Colonne Adresse :</b> Numéro et nom de rue</li>
            <li><b>Colonne Code Postal :</b> Code postal à 5 chiffres</li>
            <li><b>Colonne Ville :</b> Nom de la commune</li>
        </ul>
        <p style="margin-top: 1rem; color: #666;">
            <i>💡 Le système détectera automatiquement les colonnes et identifiera les adresses hors secteur</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="margin: 0; font-size: 1.1rem;">🚛 Optimisateur de Tournées Logistiques</p>
    <p style="margin: 0.5rem 0; opacity: 0.9;">Solution 100% gratuite • Géolocalisation automatique • Export Excel</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Par l'alternant Delestret Kim</p>
</div>
""", unsafe_allow_html=True)
