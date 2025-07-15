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
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Configuration de la page
st.set_page_config(
    page_title="Optimisateur de Tournées Logistiques",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialisation du session state avec valeurs par défaut
def init_session_state():
    defaults = {
        'geocode_cache': {},
        'current_file_hash': None,
        'processing_status': 'idle',  # idle, processing, completed
        'results': {},
        'error_message': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# CSS optimisé
st.markdown("""
<style>
    /* Performance: utiliser transform au lieu de box-shadow pour les animations */
    .main { background-color: #f5f7fa; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        will-change: transform;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s ease;
        will-change: transform;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    .download-ready {
        background: #48bb78;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .footer {
        background: #2d3748;
        color: white;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 3rem;
    }
    
    /* Désactiver les animations inutiles */
    * {
        animation-duration: 0.3s !important;
        transition-duration: 0.2s !important;
    }
    
    .tour-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="main-header">
    <h1>🚛 Optimisateur de Tournées Logistiques</h1>
    <p>Solution intelligente pour l'optimisation automatique de vos itinéraires de livraison</p>
</div>
""", unsafe_allow_html=True)

# Cache pour le géocodage avec persistance
@st.cache_data(persist=True, ttl=86400)  # Cache pour 24h
def get_cached_geocoding():
    return {}

def save_to_cache(address, result):
    cache = get_cached_geocoding()
    cache[address] = result
    return cache

# Fonction de géocodage optimisée
def geocode_address_optimized(address, geolocator=None):
    """Géocode avec cache persistant"""
    # Vérifier le cache global
    cache = get_cached_geocoding()
    if address in cache:
        return cache[address]
    
    # Vérifier le cache de session
    if address in st.session_state.geocode_cache:
        return st.session_state.geocode_cache[address]
    
    # Géocoder si pas en cache
    if geolocator is None:
        geolocator = Nominatim(user_agent="logistics_optimizer_pro", timeout=10)
    
    try:
        location = geolocator.geocode(address, exactly_one=True)
        if location:
            result = (location.latitude, location.longitude, True)
        else:
            result = (None, None, False)
    except Exception:
        result = (None, None, False)
    
    # Sauvegarder dans les deux caches
    st.session_state.geocode_cache[address] = result
    save_to_cache(address, result)
    
    return result

# Géocodage par batch avec threading
def batch_geocode_parallel(addresses, progress_bar=None, status_text=None):
    """Géocodage parallèle pour améliorer les performances"""
    results = [None] * len(addresses)
    completed = 0
    lock = threading.Lock()
    
    # Créer un geolocator par thread
    def geocode_single(idx, address):
        nonlocal completed
        geolocator = Nominatim(user_agent=f"logistics_optimizer_thread_{idx}", timeout=10)
        result = geocode_address_optimized(address, geolocator)
        
        with lock:
            results[idx] = result
            completed += 1
            if progress_bar and status_text:
                progress = completed / len(addresses)
                progress_bar.progress(progress)
                status_text.text(f"Géolocalisation : {completed}/{len(addresses)} adresses")
        
        # Pause seulement pour les nouvelles requêtes
        if address not in st.session_state.geocode_cache:
            time.sleep(1.0)  # Respecter les limites API
        
        return result
    
    # Utiliser ThreadPoolExecutor pour paralléliser
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for idx, address in enumerate(addresses):
            future = executor.submit(geocode_single, idx, address)
            futures.append(future)
        
        # Attendre que tout soit terminé
        for future in futures:
            future.result()
    
    return results

# Détection intelligente des colonnes
def detect_columns_smart(df):
    """Détection améliorée des colonnes"""
    columns = df.columns.tolist()
    column_lower = [col.lower() for col in columns]
    
    # Patterns de détection
    patterns = {
        'address': ['adresse', 'address', 'rue', 'street', 'voie', 'client', 'nom', 'lieu', 'livraison'],
        'postal': ['postal', 'cp', 'code', 'zip', 'postcode', 'code_postal'],
        'city': ['ville', 'city', 'commune', 'localite', 'locality', 'municipalite']
    }
    
    detected = {}
    
    for col_type, keywords in patterns.items():
        for i, col in enumerate(column_lower):
            if any(keyword in col for keyword in keywords):
                detected[col_type] = columns[i]
                break
    
    return detected.get('address'), detected.get('postal'), detected.get('city')

# Algorithme optimisé pour la détection des hors secteur
def detect_outliers_advanced(df_points):
    """Détection avancée des outliers avec clustering"""
    if len(df_points) < 3:
        return df_points.index.tolist(), []
    
    # Calculer le centre médian
    center_lat = df_points['lat'].median()
    center_lon = df_points['lon'].median()
    
    # Calculer toutes les distances
    distances = []
    for idx, row in df_points.iterrows():
        dist = geodesic((center_lat, center_lon), (row['lat'], row['lon'])).kilometers
        distances.append((idx, dist))
    
    # Trier par distance
    distances.sort(key=lambda x: x[1])
    
    # Utiliser une approche percentile pour identifier les outliers
    if len(distances) > 10:
        # 90e percentile comme seuil
        threshold_idx = int(len(distances) * 0.9)
        threshold_dist = distances[threshold_idx][1]
        
        # S'assurer d'un seuil minimum raisonnable
        threshold_dist = max(threshold_dist, 15.0)
    else:
        # Pour peu de points, utiliser 2x la distance médiane
        median_dist = distances[len(distances)//2][1]
        threshold_dist = max(median_dist * 2, 10.0)
    
    # Classification
    in_sector = []
    out_sector = []
    
    for idx, dist in distances:
        if dist <= threshold_dist:
            in_sector.append(idx)
        else:
            out_sector.append(idx)
    
    # Ajouter la distance au dataframe
    dist_dict = {idx: dist for idx, dist in distances}
    df_points['distance_from_center'] = df_points.index.map(dist_dict)
    
    return in_sector, out_sector

# Division en tournées optimisée
def create_multiple_tours(df_sector, max_per_tour=50):
    """Création de tournées multiples avec clustering géographique"""
    if len(df_sector) <= max_per_tour:
        return [df_sector]
    
    n_tours = math.ceil(len(df_sector) / max_per_tour)
    
    # K-means simplifié pour le clustering géographique
    tours = []
    points = df_sector[['lat', 'lon']].values
    
    # Initialiser les centres avec k-means++
    centers = []
    centers.append(points[np.random.randint(len(points))])
    
    for _ in range(1, n_tours):
        distances = []
        for point in points:
            min_dist = min([np.sqrt(np.sum((point - center)**2)) for center in centers])
            distances.append(min_dist)
        
        probabilities = np.array(distances) / sum(distances)
        centers.append(points[np.random.choice(len(points), p=probabilities)])
    
    # Assigner chaque point au centre le plus proche
    assignments = []
    for point in points:
        distances = [np.sqrt(np.sum((point - center)**2)) for center in centers]
        assignments.append(np.argmin(distances))
    
    # Créer les tournées
    for i in range(n_tours):
        tour_indices = [idx for idx, assignment in enumerate(assignments) if assignment == i]
        if tour_indices:
            tour_df = df_sector.iloc[tour_indices]
            tours.append(tour_df)
    
    return tours

# Optimisation TSP améliorée
def optimize_route_2opt(df_points):
    """Algorithme 2-opt pour l'optimisation de route"""
    if len(df_points) <= 2:
        return df_points.index.tolist()
    
    # Matrice de distances pré-calculée
    n = len(df_points)
    dist_matrix = np.zeros((n, n))
    points = df_points[['lat', 'lon']].values
    
    for i in range(n):
        for j in range(i+1, n):
            dist = geodesic(points[i], points[j]).kilometers
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    # Tour initial avec nearest neighbor
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # Amélioration 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                
                # Calculer le gain
                gain = (dist_matrix[tour[i-1]][tour[i]] + dist_matrix[tour[j-1]][tour[j]] -
                       dist_matrix[tour[i-1]][tour[j-1]] - dist_matrix[tour[i]][tour[j]])
                
                if gain > 0:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
                    break
            if improved:
                break
    
    # Convertir en indices du dataframe
    return [df_points.index[i] for i in tour]

# Création de carte optimisée
def create_optimized_map(all_tours, df_out_sector=None):
    """Création de carte avec performances optimisées"""
    # Calculer le centre global
    all_points = []
    for tour in all_tours:
        all_points.extend(tour[['lat', 'lon']].values.tolist())
    
    if not all_points:
        return None
    
    center = np.mean(all_points, axis=0)
    
    # Créer la carte
    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles='OpenStreetMap',
        prefer_canvas=True  # Meilleure performance pour beaucoup de marqueurs
    )
    
    # Couleurs pour les tournées
    colors = ['blue', 'green', 'purple', 'darkred', 'orange', 'cadetblue', 'darkgreen']
    
    # Ajouter les tournées
    for tour_idx, tour_df in enumerate(all_tours):
        color = colors[tour_idx % len(colors)]
        
        # Points et route
        coords = tour_df[['lat', 'lon']].values.tolist()
        
        # Ligne de la tournée
        folium.PolyLine(
            coords,
            color=color,
            weight=3,
            opacity=0.8
        ).add_to(m)
        
        # Marqueurs principaux seulement (début et fin)
        folium.Marker(
            location=coords[0],
            popup=f"Tournée {tour_idx + 1} - Départ",
            icon=folium.Icon(color=color, icon='play', prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            location=coords[-1],
            popup=f"Tournée {tour_idx + 1} - Arrivée",
            icon=folium.Icon(color=color, icon='stop', prefix='fa')
        ).add_to(m)
        
        # Points intermédiaires en cercles pour performance
        for i, coord in enumerate(coords[1:-1], 1):
            folium.CircleMarker(
                location=coord,
                radius=5,
                popup=f"T{tour_idx + 1}-{i+1}",
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
    
    # Hors secteur
    if df_out_sector is not None and len(df_out_sector) > 0:
        for _, row in df_out_sector.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup="Hors secteur",
                icon=folium.Icon(color='red', icon='exclamation', prefix='fa')
            ).add_to(m)
    
    return m

# Génération Excel optimisée
def generate_excel_optimized(all_tours, df_out_sector, df_failed, address_col, postal_col, city_col):
    """Génération Excel avec structure optimisée"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Synthèse globale en première page
        summary_data = []
        total_distance_global = 0
        total_addresses = 0
        
        for idx, tour in enumerate(all_tours):
            dist = tour['distance_etape'].sum() if 'distance_etape' in tour.columns else 0
            total_distance_global += dist
            total_addresses += len(tour)
            
            summary_data.append({
                'Tournée': idx + 1,
                'Adresses': len(tour),
                'Distance (km)': round(dist, 1),
                'Temps estimé': f"{int((dist*3 + len(tour)*5)//60)}h{int((dist*3 + len(tour)*5)%60)}min"
            })
        
        # Page synthèse
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Synthèse', index=False)
        
        # Une feuille par tournée
        for idx, tour in enumerate(all_tours):
            sheet_name = f'Tournée_{idx + 1}'
            
            # Préparer les données
            export_df = tour.copy()
            export_df.insert(0, 'Ordre', range(1, len(export_df) + 1))
            
            # Colonnes essentielles
            cols = ['Ordre', address_col, postal_col, city_col]
            
            if 'distance_etape' in export_df.columns:
                export_df['Distance (km)'] = export_df['distance_etape'].round(1)
                cols.append('Distance (km)')
            
            # Coordonnées GPS
            export_df['GPS'] = export_df.apply(lambda row: f"{row['lat']:.6f}, {row['lon']:.6f}", axis=1)
            cols.append('GPS')
            
            export_df[cols].to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Hors secteur
        if len(df_out_sector) > 0:
            out_df = df_out_sector[[address_col, postal_col, city_col, 'distance_from_center']].copy()
            out_df['Distance centre (km)'] = out_df['distance_from_center'].round(1)
            out_df.drop('distance_from_center', axis=1, inplace=True)
            out_df.to_excel(writer, sheet_name='Hors_Secteur', index=False)
        
        # Échecs
        if len(df_failed) > 0:
            df_failed[[address_col, postal_col, city_col]].to_excel(
                writer, sheet_name='Échecs', index=False
            )
    
    output.seek(0)
    return output

# Fonction principale de traitement
def process_addresses(df, address_col, postal_col, city_col, progress_container):
    """Traitement principal optimisé"""
    # Nettoyage des données
    df_clean = df.dropna(subset=[address_col, postal_col, city_col]).copy()
    df_clean[postal_col] = df_clean[postal_col].astype(str).str.extract('(\d{5})')[0]
    df_clean = df_clean.dropna(subset=[postal_col])
    
    # Limiter à 250 adresses
    if len(df_clean) > 250:
        df_clean = df_clean.head(250)
        st.warning("⚠️ Limitation à 250 adresses appliquée")
    
    # Construction des adresses complètes
    df_clean['adresse_complete'] = (
        df_clean[address_col].astype(str).str.strip() + ", " + 
        df_clean[postal_col].astype(str) + " " + 
        df_clean[city_col].astype(str).str.strip() + ", France"
    )
    
    # Géocodage
    with progress_container.container():
        st.markdown("### 🌍 Géolocalisation des adresses")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Géocodage parallèle
        coordinates = batch_geocode_parallel(
            df_clean['adresse_complete'].tolist(),
            progress_bar,
            status_text
        )
    
    # Ajouter les résultats
    df_clean[['lat', 'lon', 'geocoding_success']] = pd.DataFrame(coordinates)
    
    # Séparer succès/échecs
    df_geocoded = df_clean[df_clean['geocoding_success'] == True].copy()
    df_failed = df_clean[df_clean['geocoding_success'] == False].copy()
    
    if len(df_geocoded) < 2:
        raise ValueError("Pas assez d'adresses géolocalisées pour créer un itinéraire")
    
    # Détection des hors secteur
    in_sector_idx, out_sector_idx = detect_outliers_advanced(df_geocoded)
    df_sector = df_geocoded.loc[in_sector_idx].copy()
    df_out_sector = df_geocoded.loc[out_sector_idx].copy()
    
    if len(df_sector) < 2:
        raise ValueError("Pas assez d'adresses dans le secteur principal")
    
    # Création des tournées
    with progress_container.container():
        st.markdown("### 🚛 Optimisation des tournées")
        progress_bar2 = st.progress(0)
        
        tours = create_multiple_tours(df_sector, max_per_tour=50)
        all_tours_optimized = []
        
        for i, tour in enumerate(tours):
            progress_bar2.progress((i + 1) / len(tours))
            
            # Optimiser chaque tournée
            optimal_order = optimize_route_2opt(tour)
            tour_optimized = tour.loc[optimal_order].reset_index(drop=True)
            
            # Calculer les distances
            distances = [0]
            for j in range(1, len(tour_optimized)):
                coord1 = (tour_optimized.iloc[j-1]['lat'], tour_optimized.iloc[j-1]['lon'])
                coord2 = (tour_optimized.iloc[j]['lat'], tour_optimized.iloc[j]['lon'])
                distances.append(geodesic(coord1, coord2).kilometers)
            
            tour_optimized['distance_etape'] = distances
            all_tours_optimized.append(tour_optimized)
    
    return all_tours_optimized, df_out_sector, df_failed

# Interface principale
def main():
    # Zone de fichier
    st.markdown("### 📁 Importez votre fichier Excel")
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier (max 250 adresses)",
        type=["xlsx", "xls"],
        help="Format : colonnes Adresse, Code Postal et Ville",
        key="file_uploader"
    )
    
    # Container pour les résultats
    results_container = st.container()
    progress_container = st.empty()
    
    # Si des résultats existent, les afficher en premier
    if st.session_state.processing_status == 'completed' and 'results' in st.session_state:
        with results_container:
            results = st.session_state.results
            
            # Zone de téléchargement
            st.markdown('<div class="download-ready">', unsafe_allow_html=True)
            st.markdown("### ✅ Vos tournées sont prêtes !")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.download_button(
                    label="📥 **Télécharger le fichier Excel**",
                    data=results['excel_data'],
                    file_name=f"tournees_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type="primary",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Nouveau", type="secondary"):
                    st.session_state.processing_status = 'idle'
                    st.session_state.results = {}
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistiques
            st.markdown("### 📊 Statistiques")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚛 Tournées", results['n_tours'])
            with col2:
                st.metric("📦 Adresses", results['n_addresses'])
            with col3:
                st.metric("📏 Distance totale", f"{results['total_distance']:.1f} km")
            with col4:
                st.metric("⚠️ Hors secteur", results['n_out_sector'])
            
            # Détails par tournée
            st.markdown("### 🗺️ Détails des tournées")
            
            for idx, tour in enumerate(results['tours']):
                tour_dist = tour['distance_etape'].sum()
                tour_time = tour_dist * 3 + len(tour) * 5
                
                with st.expander(f"Tournée {idx + 1} - {len(tour)} adresses - {tour_dist:.1f} km", expanded=(idx == 0)):
                    # Afficher les premières adresses
                    display_cols = [col for col in tour.columns if any(
                        kw in col.lower() for kw in ['adresse', 'postal', 'ville']
                    )]
                    
                    if display_cols:
                        st.dataframe(
                            tour[display_cols].head(10),
                            use_container_width=True,
                            hide_index=True
                        )
                        if len(tour) > 10:
                            st.info(f"... et {len(tour) - 10} autres adresses")
            
            # Carte
            if 'map' in results and results['map']:
                st.markdown("### 🗺️ Visualisation")
                st_folium(results['map'], height=600, use_container_width=True)
    
    # Traitement du fichier
    if uploaded_file and st.session_state.processing_status != 'processing':
        # Calculer le hash du fichier
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Vérifier si c'est un nouveau fichier
        if file_hash != st.session_state.current_file_hash:
            st.session_state.current_file_hash = file_hash
            st.session_state.processing_status = 'idle'
            st.session_state.results = {}
        
        # Lire le fichier
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Fichier chargé : {len(df)} adresses")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            return
        
        # Détection des colonnes
        address_col, postal_col, city_col = detect_columns_smart(df)
        
        # Interface de sélection si nécessaire
        col1, col2, col3 = st.columns(3)
        with col1:
            address_col = st.selectbox(
                "📍 Colonne Adresse",
                df.columns.tolist(),
                index=df.columns.tolist().index(address_col) if address_col else 0
            )
        with col2:
            postal_col = st.selectbox(
                "📮 Code Postal",
                df.columns.tolist(),
                index=df.columns.tolist().index(postal_col) if postal_col else 0
            )
        with col3:
            city_col = st.selectbox(
                "🏙️ Ville",
                df.columns.tolist(),
                index=df.columns.tolist().index(city_col) if city_col else 0
            )
        
        # Bouton de traitement
        if st.button("🚀 **Optimiser les tournées**", type="primary", use_container_width=True):
            st.session_state.processing_status = 'processing'
            
            try:
                # Traitement principal
                all_tours, df_out_sector, df_failed = process_addresses(
                    df, address_col, postal_col, city_col, progress_container
                )
                
                # Générer les résultats
                with progress_container.container():
                    st.info("📊 Génération des fichiers...")
                    
                    # Excel
                    excel_data = generate_excel_optimized(
                        all_tours, df_out_sector, df_failed,
                        address_col, postal_col, city_col
                    )
                    
                    # Carte
                    map_data = create_optimized_map(all_tours, df_out_sector)
                    
                    # Calculer les statistiques
                    total_distance = sum(tour['distance_etape'].sum() for tour in all_tours)
                    total_addresses = sum(len(tour) for tour in all_tours)
                
                # Sauvegarder les résultats
                st.session_state.results = {
                    'tours': all_tours,
                    'excel_data': excel_data,
                    'map': map_data,
                    'n_tours': len(all_tours),
                    'n_addresses': total_addresses,
                    'n_out_sector': len(df_out_sector),
                    'total_distance': total_distance
                }
                
                st.session_state.processing_status = 'completed'
                
                # Rafraîchir pour afficher les résultats
                st.rerun()
                
            except Exception as e:
                st.session_state.processing_status = 'idle'
                st.session_state.error_message = str(e)
                st.error(f"❌ Erreur : {str(e)}")
    
    # Instructions si pas de fichier
    elif not uploaded_file and st.session_state.processing_status == 'idle':
        st.info("""
        ### 📋 Comment utiliser l'application :
        
        1. **Importez** votre fichier Excel (max 250 adresses)
        2. **Vérifiez** que les colonnes sont correctement détectées
        3. **Cliquez** sur "Optimiser les tournées"
        4. **Téléchargez** le fichier Excel avec les itinéraires
        
        **Format requis :** Adresse, Code Postal, Ville
        """)

# Footer
st.markdown("""
<div class="footer">
    <p style="margin: 0; font-size: 1.1rem;">🚛 Optimisateur de Tournées Logistiques</p>
    <p style="margin: 0.5rem 0; opacity: 0.9;">Solution 100% gratuite • Jusqu'à 250 adresses • Performance optimisée</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Par l'alternant Delestret Kim</p>
</div>
""", unsafe_allow_html=True)

# Lancer l'application
if __name__ == "__main__":
    main()
