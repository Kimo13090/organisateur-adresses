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
    
    /* Tables */
    .dataframe {
        border: none !important;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Alertes personnalisées */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
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
    
    /* Badges pour les statuts */
    .badge-success {
        background-color: #48bb78;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .badge-warning {
        background-color: #ed8936;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
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
    """
    Détecte automatiquement les adresses hors secteur en utilisant la Médiane Absolute Deviation (MAD)
    Plus robuste que l'écart-type pour détecter les valeurs aberrantes
    """
    if len(df_points) < 5:
        return df_points.index.tolist(), []
    
    # Calculer le centre géographique médian
    center_lat = df_points['lat'].median()
    center_lon = df_points['lon'].median()
    center = (center_lat, center_lon)
    
    # Calculer les distances depuis le centre médian
    distances = []
    for idx, row in df_points.iterrows():
        point = (row['lat'], row['lon'])
        dist = geodesic(center, point).kilometers
        distances.append(dist)
    
    df_points['distance_from_center'] = distances
    
    # Calcul de la MAD (Median Absolute Deviation)
    median_distance = np.median(distances)
    mad = np.median(np.abs(distances - median_distance))
    
    # Seuil adaptatif basé sur la MAD
    # Le facteur 2.5 est un compromis pour détecter les vraies anomalies
    threshold = median_distance + 2.5 * mad
    
    # S'assurer d'avoir un seuil minimum raisonnable
    threshold = max(threshold, 10.0)  # Au minimum 10 km
    
    # Classification
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
    # Centre de la carte
    center_lat = df_route['lat'].mean()
    center_lon = df_route['lon'].mean()
    
    # Création de la carte
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Ajouter les points de l'itinéraire principal
    for i, row in df_route.iterrows():
        # Couleur selon la position
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
            <i>Distance depuis précédent: {row.get('distance_etape', 'N/A')}</i>
            """,
            tooltip=f"Stop {i+1}",
            icon=folium.Icon(color=color, icon=icon, prefix=prefix)
        ).add_to(m)
    
    # Tracer l'itinéraire
    route_coords = [[row['lat'], row['lon']] for _, row in df_route.iterrows()]
    folium.PolyLine(
        route_coords,
        color='purple',
        weight=3,
        opacity=0.8,
        smooth_factor=2
    ).add_to(m)
    
    # Ajouter les points hors secteur si présents
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
    
    # Ajouter une légende
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
    
    # Trouver le point de départ optimal (centre de gravité)
    center_lat = df_points['lat'].mean()
    center_lon = df_points['lon'].mean()
    
    # Trouver le point le plus proche du centre
    min_dist = float('inf')
    start_idx = df_points.index[0]
    
    for idx, row in df_points.iterrows():
        dist = geodesic((center_lat, center_lon), (row['lat'], row['lon'])).kilometers
        if dist < min_dist:
            min_dist = dist
            start_idx = idx
    
    # Algorithme du plus proche voisin
    route = [start_idx]
    unvisited = set(df_points.index) - {start_idx}
    
    while unvisited:
        current_idx = route[-1]
        current_coords = (df_points.at[current_idx, 'lat'], df_points.at[current_idx, 'lon'])
        
        # Trouver le plus proche non visité
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

# Interface principale
st.markdown("### 📁 Importez votre fichier Excel")

uploaded_file = st.file_uploader(
    "Glissez-déposez votre fichier contenant les adresses clients",
    type=["xlsx", "xls"],
    help="Format requis : colonnes Adresse, Code Postal et Ville"
)

if not uploaded_file:
    # Message d'attente stylisé
    st.markdown("""
    <div class="status-card">
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
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Par l'alternant Delestret Kim</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# Lecture du fichier
try:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ Fichier importé avec succès : **{len(df)} adresses** détectées")
except Exception as e:
    st.error(f"❌ Erreur lors de la lecture : {e}")
    st.stop()

# Aperçu des données
with st.expander("📊 Aperçu des données", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

# Détection automatique des colonnes
address_col, postal_col, city_col = detect_columns_smart(df)

if not all([address_col, postal_col, city_col]):
    st.warning("⚠️ Sélection manuelle des colonnes requise")
    
    col1, col2, col3 = st.columns(3)
    columns = df.columns.tolist()
    
    with col1:
        address_col = st.selectbox("📍 Colonne Adresse", columns)
    with col2:
        postal_col = st.selectbox("📮 Code Postal", columns)
    with col3:
        city_col = st.selectbox("🏙️ Ville", columns)
else:
    st.info(f"✅ Colonnes détectées : **{address_col}**, **{postal_col}**, **{city_col}**")

# Nettoyage automatique des données
df_clean = df.dropna(subset=[address_col, postal_col, city_col]).copy()
df_clean[postal_col] = df_clean[postal_col].astype(str).str.extract('(\d+)')[0]
df_clean = df_clean[df_clean[address_col].astype(str).str.strip() != '']
df_clean = df_clean[df_clean[city_col].astype(str).str.strip() != '']

# Bouton de traitement
if st.button("🚀 **Générer l'itinéraire optimisé**", type="primary", use_container_width=True):
    
    # Construction des adresses complètes
    df_clean['adresse_complete'] = (
        df_clean[address_col].astype(str) + ", " + 
        df_clean[postal_col].astype(str) + " " + 
        df_clean[city_col].astype(str) + ", France"
    )
    
    # Section géocodage
    st.markdown("---")
    st.markdown("### 🌍 Géolocalisation des adresses")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    coordinates = []
    geocoding_success = []
    total = len(df_clean)
    
    # Géocodage avec limitation pour rester gratuit
    for i, address in enumerate(df_clean['adresse_complete']):
        if i >= 100:  # Limite pour éviter les quotas
            st.warning("⚠️ Limitation à 100 adresses pour respecter les quotas gratuits")
            df_clean = df_clean.head(100)
            break
            
        progress = (i + 1) / min(total, 100)
        progress_bar.progress(progress)
        status_text.text(f"Traitement : {i+1}/{min(total, 100)} - {address[:50]}...")
        
        lat, lon, success = geocode_address(address)
        coordinates.append((lat, lon))
        geocoding_success.append(success)
        
        time.sleep(1.1)  # Respecter les limites de l'API gratuite
    
    df_clean = df_clean.head(len(coordinates))
    df_clean[['lat', 'lon']] = pd.DataFrame(coordinates)
    df_clean['geocoding_success'] = geocoding_success
    
    # Résultats du géocodage
    df_geocoded = df_clean[df_clean['geocoding_success'] == True].dropna(subset=['lat', 'lon']).copy()
    df_failed = df_clean[df_clean['geocoding_success'] == False].copy()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Géolocalisées", len(df_geocoded))
    with col2:
        st.metric("❌ Échecs", len(df_failed))
    with col3:
        st.metric("📊 Taux de succès", f"{len(df_geocoded)/len(df_clean)*100:.1f}%")
    
    if len(df_geocoded) < 2:
        st.error("❌ Pas assez d'adresses géolocalisées")
        st.stop()
    
    # Détection automatique des adresses hors secteur
    st.markdown("### 🎯 Analyse du secteur de livraison")
    
    with st.spinner("Détection automatique des adresses hors secteur..."):
        in_sector_idx, out_sector_idx = find_outliers_using_mad(df_geocoded)
    
    df_sector = df_geocoded.loc[in_sector_idx].copy()
    df_out_sector = df_geocoded.loc[out_sector_idx].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Dans le secteur principal", len(df_sector))
    with col2:
        st.metric("⚠️ Hors secteur", len(df_out_sector))
    
    if len(df_out_sector) > 0:
        with st.expander("🚨 Adresses hors secteur détectées", expanded=True):
            df_out_display = df_out_sector[[address_col, postal_col, city_col, 'distance_from_center']].copy()
            df_out_display['distance_from_center'] = df_out_display['distance_from_center'].round(1).astype(str) + ' km'
            st.dataframe(df_out_display, use_container_width=True)
    
    # Optimisation de la tournée
    st.markdown("### 🛣️ Optimisation de l'itinéraire")
    
    with st.spinner("Calcul de l'itinéraire optimal..."):
        optimal_order = optimize_route_nearest_neighbor(df_sector)
        df_optimized = df_sector.loc[optimal_order].reset_index(drop=True)
        df_optimized, total_distance = calculate_route_distances(df_optimized)
    
    # Statistiques
    estimated_time = total_distance * 3 + len(df_optimized) * 5
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Livraisons", len(df_optimized))
    with col2:
        st.metric("📏 Distance totale", f"{total_distance:.1f} km")
    with col3:
        st.metric("⏱️ Temps estimé", f"{int(estimated_time//60)}h {int(estimated_time%60)}min")
    with col4:
        st.metric("⚡ Vitesse moyenne", f"{total_distance/(estimated_time/60):.1f} km/h")
    
    # Carte interactive
    st.markdown("### 🗺️ Visualisation de la tournée")
    
    with st.spinner("Génération de la carte interactive..."):
        folium_map = create_folium_map(df_optimized, df_out_sector)
        st_folium(folium_map, height=500, use_container_width=True)
    
    # Tableau de l'itinéraire
    st.markdown("### 📋 Détail de l'itinéraire optimisé")
    
    df_display = df_optimized.copy()
    df_display.insert(0, 'Ordre', range(1, len(df_display) + 1))
    df_display['Distance étape'] = df_display['distance_etape'].round(1).astype(str) + ' km'
    df_display['Heure estimée'] = pd.to_datetime('08:00:00') + pd.to_timedelta(df_display.index * 5 + df_display['distance_cumulee'] * 3, unit='minutes')
    df_display['Heure estimée'] = df_display['Heure estimée'].dt.strftime('%H:%M')
    
    # Ajout du statut
    df_display['Statut'] = '<span class="badge-success">Dans secteur</span>'
    
    display_cols = ['Ordre', address_col, postal_col, city_col, 'Distance étape', 'Heure estimée']
    st.dataframe(
        df_display[display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    # Export Excel amélioré
    st.markdown("### 💾 Téléchargement des résultats")
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille principale avec indicateur de secteur
        export_main = df_display.copy()
        export_main['Latitude'] = df_optimized['lat']
        export_main['Longitude'] = df_optimized['lon']
        export_main['Secteur'] = 'Principal'
        export_main.to_excel(writer, index=False, sheet_name='Itinéraire_Optimisé')
        
        # Feuille des adresses hors secteur
        if len(df_out_sector) > 0:
            export_out = df_out_sector.copy()
            export_out['Secteur'] = 'Hors secteur'
            export_out['Distance du centre (km)'] = export_out['distance_from_center'].round(1)
            export_out.to_excel(writer, index=False, sheet_name='Hors_Secteur')
        
        # Feuille des échecs de géocodage
        if len(df_failed) > 0:
            df_failed.to_excel(writer, index=False, sheet_name='Échecs_Géolocalisation')
        
        # Feuille de synthèse
        summary_data = {
            'Métrique': ['Total adresses', 'Adresses géolocalisées', 'Dans le secteur', 'Hors secteur', 
                        'Distance totale (km)', 'Temps estimé (min)', 'Heure de départ', 'Heure d\'arrivée'],
            'Valeur': [len(df_clean), len(df_geocoded), len(df_sector), len(df_out_sector),
                      round(total_distance, 1), int(estimated_time), '08:00',
                      (pd.to_datetime('08:00:00') + pd.to_timedelta(estimated_time, unit='minutes')).strftime('%H:%M')]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Synthèse')
    
    output.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"tournee_optimisee_{timestamp}.xlsx"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.download_button(
            label="📥 **Télécharger le fichier Excel complet**",
            data=output,
            file_name=filename,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            type="primary",
            use_container_width=True
        )
    
    # Message de succès
    st.success("✅ **Tournée optimisée avec succès !**")
    st.balloons()
    
    # Résumé
    st.markdown(f"""
    <div class="status-card">
        <h4>📊 Résumé de la tournée</h4>
        <ul>
            <li>🚛 <b>{len(df_optimized)} livraisons</b> dans le secteur principal</li>
            <li>📏 Distance totale : <b>{total_distance:.1f} km</b></li>
            <li>⏱️ Durée estimée : <b>{int(estimated_time//60)}h {int(estimated_time%60)}min</b></li>
            <li>🚨 <b>{len(df_out_sector)} adresses hors secteur</b> identifiées automatiquement</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer permanent
st.markdown("""
<div class="footer">
    <p style="margin: 0; font-size: 1.1rem;">🚛 Optimisateur de Tournées Logistiques</p>
    <p style="margin: 0.5rem 0; opacity: 0.9;">Solution 100% gratuite • Géolocalisation automatique • Export Excel</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Par l'alternant Delestret Kim</p>
</div>
""", unsafe_allow_html=True)
