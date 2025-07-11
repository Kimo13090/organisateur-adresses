import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import time
import numpy as np
from datetime import datetime
import math

# Gestion des imports optionnels
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="Organisateur de Tournées Automatique",
    page_icon="🚛",
    layout="wide"
)

# Titre principal
st.title("🚛 Organisateur de Tournées Automatique")
st.markdown("*Optimisation intelligente pour itinéraires de livraison*")
st.markdown("---")

@st.cache_data
def geocode_address(address, max_retries=3):
    """Géocode une adresse avec retry et gestion d'erreurs"""
    geolocator = Nominatim(user_agent="delivery_route_optimizer")
    
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
            time.sleep(2)  # Pause plus longue entre tentatives
    
    return (None, None, False)

def detect_outliers_automatic(df_points):
    """Détection automatique des points aberrants avec DBSCAN ou méthode alternative"""
    if len(df_points) < 5:
        return df_points, pd.DataFrame()
    
    # Utiliser DBSCAN si sklearn est disponible
    if SKLEARN_AVAILABLE:
        # Préparation des données pour clustering
        coordinates = df_points[['lat', 'lon']].values
        
        # Calcul automatique des paramètres DBSCAN
        # eps basé sur la distance moyenne entre points
        distances = []
        for i in range(len(coordinates)):
            for j in range(i+1, len(coordinates)):
                dist = geodesic(coordinates[i], coordinates[j]).kilometers
                distances.append(dist)
        
        if distances:
            avg_distance = np.mean(distances)
            # eps = distance moyenne * facteur (plus conservateur pour livraisons)
            eps = avg_distance * 0.7
            min_samples = max(2, int(len(df_points) * 0.1))  # 10% minimum des points
            
            # Application DBSCAN
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coordinates)
            
            clustering = DBSCAN(eps=eps/100, min_samples=min_samples).fit(coords_scaled)
            labels = clustering.labels_
            
            # Identification du cluster principal (le plus grand)
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            
            if len(unique_labels) > 0:
                main_cluster = unique_labels[np.argmax(counts)]
                
                # Séparation des points
                df_main = df_points[labels == main_cluster].copy()
                df_outliers = df_points[labels != main_cluster].copy()
                
                return df_main, df_outliers
    
    # Si pas assez de données pour clustering ou sklearn non disponible, utiliser méthode distance
    return fallback_outlier_detection(df_points)

def fallback_outlier_detection(df_points):
    """Méthode de fallback pour détecter les aberrants"""
    if len(df_points) < 3:
        return df_points, pd.DataFrame()
    
    # Calcul du centroïde
    centroid = (df_points['lat'].mean(), df_points['lon'].mean())
    
    # Calcul des distances au centroïde
    distances = []
    for idx, row in df_points.iterrows():
        dist = geodesic((row['lat'], row['lon']), centroid).kilometers
        distances.append(dist)
    
    df_points['distance_centroid'] = distances
    
    # Détection automatique du seuil (méthode IQR)
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    
    # Séparation
    df_main = df_points[df_points['distance_centroid'] <= threshold].copy()
    df_outliers = df_points[df_points['distance_centroid'] > threshold].copy()
    
    return df_main, df_outliers

def optimize_delivery_route(df_points):
    """Optimisation de tournée adaptée aux livraisons"""
    if len(df_points) <= 1:
        return df_points.index.tolist()
    
    # Étape 1: Trouver le point de départ optimal (plus excentré)
    # Pour les livraisons, on commence souvent par la périphérie
    center_lat = df_points['lat'].mean()
    center_lon = df_points['lon'].mean()
    
    # Calcul des distances au centre et choix du point le plus excentré
    distances_to_center = []
    for idx, row in df_points.iterrows():
        dist = geodesic((row['lat'], row['lon']), (center_lat, center_lon)).kilometers
        distances_to_center.append((idx, dist))
    
    # Commencer par un point intermédiaire (ni trop central, ni trop excentré)
    distances_to_center.sort(key=lambda x: x[1])
    start_idx = distances_to_center[len(distances_to_center)//3][0]  # 1/3 du chemin
    
    # Étape 2: Algorithme du plus proche voisin avec optimisation 2-opt
    route = [start_idx]
    remaining = set(df_points.index) - {start_idx}
    
    current_point = start_idx
    while remaining:
        current_coords = (df_points.at[current_point, 'lat'], df_points.at[current_point, 'lon'])
        
        # Trouver le point le plus proche
        min_distance = float('inf')
        next_point = None
        
        for point in remaining:
            point_coords = (df_points.at[point, 'lat'], df_points.at[point, 'lon'])
            distance = geodesic(current_coords, point_coords).kilometers
            
            if distance < min_distance:
                min_distance = distance
                next_point = point
        
        route.append(next_point)
        remaining.remove(next_point)
        current_point = next_point
    
    # Étape 3: Amélioration avec 2-opt (échange de segments)
    route = improve_route_2opt(df_points, route)
    
    return route

def improve_route_2opt(df_points, route):
    """Amélioration de la route avec l'algorithme 2-opt"""
    if len(route) < 4:
        return route
    
    def calculate_route_distance(route_order):
        total_dist = 0
        for i in range(len(route_order) - 1):
            coord1 = (df_points.at[route_order[i], 'lat'], df_points.at[route_order[i], 'lon'])
            coord2 = (df_points.at[route_order[i+1], 'lat'], df_points.at[route_order[i+1], 'lon'])
            total_dist += geodesic(coord1, coord2).kilometers
        return total_dist
    
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route)
    
    improved = True
    iterations = 0
    max_iterations = min(100, len(route) * 2)  # Limiter les itérations
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                
                # Créer nouvelle route en inversant le segment
                new_route = route.copy()
                new_route[i:j] = route[i:j][::-1]
                
                new_distance = calculate_route_distance(new_route)
                
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    route = new_route
                    improved = True
                    break
            
            if improved:
                break
    
    return best_route

def calculate_route_stats(df_route):
    """Calcul des statistiques de la tournée"""
    if len(df_route) < 2:
        return 0, 0, 0
    
    total_distance = 0
    distances = []
    
    for i in range(len(df_route) - 1):
        coord1 = (df_route.iloc[i]['lat'], df_route.iloc[i]['lon'])
        coord2 = (df_route.iloc[i + 1]['lat'], df_route.iloc[i + 1]['lon'])
        segment_distance = geodesic(coord1, coord2).kilometers
        total_distance += segment_distance
        distances.append(segment_distance)
    
    # Statistiques
    avg_distance = np.mean(distances) if distances else 0
    max_distance = max(distances) if distances else 0
    
    return total_distance, avg_distance, max_distance

# Interface utilisateur simplifiée
st.header("📁 Chargement du fichier")
uploaded_file = st.file_uploader(
    "Déposez votre fichier Excel avec les adresses clients",
    type=["xlsx"],
    help="Le fichier doit contenir les colonnes : adresse, code postal, ville"
)

if not uploaded_file:
    st.info("🔄 En attente du fichier Excel...")
    st.markdown("### 📋 Format attendu :")
    st.markdown("- **Colonne 1:** Adresse du client")
    st.markdown("- **Colonne 2:** Code postal")
    st.markdown("- **Colonne 3:** Ville")
    st.stop()

# Lecture du fichier
try:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ Fichier chargé avec succès : {len(df)} adresses")
except Exception as e:
    st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
    st.stop()

# Aperçu des données
st.markdown("---")
st.subheader("📊 Aperçu des données")
st.dataframe(df.head(10), use_container_width=True)

# Détection intelligente des colonnes
def detect_columns_smart(df):
    """Détection intelligente des colonnes adresse, code postal et ville"""
    columns = df.columns.tolist()
    
    # Mots-clés pour la détection
    address_keywords = ['adresse', 'address', 'rue', 'street', 'voie', 'avenue', 'boulevard', 'chemin', 'client']
    postal_keywords = ['postal', 'cp', 'code', 'zip', 'postcode']
    city_keywords = ['ville', 'city', 'commune', 'localite', 'locality']
    
    address_col = None
    postal_col = None
    city_col = None
    
    # Recherche par mots-clés
    for col in columns:
        col_lower = col.lower()
        
        # Détection adresse
        if not address_col:
            for keyword in address_keywords:
                if keyword in col_lower:
                    address_col = col
                    break
        
        # Détection code postal
        if not postal_col:
            for keyword in postal_keywords:
                if keyword in col_lower:
                    postal_col = col
                    break
        
        # Détection ville
        if not city_col:
            for keyword in city_keywords:
                if keyword in col_lower:
                    city_col = col
                    break
    
    return address_col, postal_col, city_col

# Détection automatique intelligente
address_col, postal_col, city_col = detect_columns_smart(df)

# Si détection automatique échoue, permettre sélection manuelle
if not all([address_col, postal_col, city_col]):
    st.warning("⚠️ Détection automatique des colonnes échouée. Sélection manuelle requise.")
    
    col1, col2, col3 = st.columns(3)
    columns = df.columns.tolist()
    
    with col1:
        address_col = st.selectbox("📍 Colonne Adresse", columns, key="manual_address")
    with col2:
        postal_col = st.selectbox("📮 Colonne Code Postal", columns, key="manual_postal")
    with col3:
        city_col = st.selectbox("🏙️ Colonne Ville", columns, key="manual_city")
    
    if not all([address_col, postal_col, city_col]):
        st.error("❌ Veuillez sélectionner toutes les colonnes requises")
        st.stop()
    
    st.info(f"📍 Colonnes sélectionnées : {address_col}, {postal_col}, {city_col}")
else:
    st.success(f"✅ Colonnes détectées automatiquement : {address_col}, {postal_col}, {city_col}")

# Validation et nettoyage des données
st.markdown("---")
st.subheader("🧹 Nettoyage des données")

# Vérification des données manquantes
missing_data = df[[address_col, postal_col, city_col]].isnull().sum()
if missing_data.any():
    st.warning("⚠️ Données manquantes détectées :")
    for col, count in missing_data.items():
        if count > 0:
            st.write(f"- {col}: {count} valeurs manquantes")

# Nettoyage des données
df_clean = df.dropna(subset=[address_col, postal_col, city_col]).copy()

# Nettoyage des codes postaux (garder seulement les chiffres)
df_clean[postal_col] = df_clean[postal_col].astype(str).str.extract('(\d+)')[0]
df_clean = df_clean.dropna(subset=[postal_col])

# Suppression des lignes avec des valeurs vides après nettoyage
df_clean = df_clean[df_clean[address_col].astype(str).str.strip() != '']
df_clean = df_clean[df_clean[city_col].astype(str).str.strip() != '']
df_clean = df_clean[df_clean[postal_col].astype(str).str.strip() != '']

st.info(f"📊 Données nettoyées : {len(df_clean)}/{len(df)} adresses valides")

if len(df_clean) == 0:
    st.error("❌ Aucune donnée valide après nettoyage")
    st.stop()

# Affichage des données nettoyées
st.dataframe(df_clean[[address_col, postal_col, city_col]].head(10), use_container_width=True)

# Limitation raisonnable pour éviter les quotas API
MAX_ADDRESSES = 200
if len(df) > MAX_ADDRESSES:
    st.warning(f"⚠️ Trop d'adresses ({len(df)}). Limitation à {MAX_ADDRESSES} pour éviter les quotas API")
    df = df.head(MAX_ADDRESSES)

# Bouton de traitement automatique
if st.button("🚀 Organiser la tournée automatiquement", type="primary", use_container_width=True):
    
    # Construction des adresses complètes
    df['adresse_complete'] = (
        df[address_col].astype(str) + ", " + 
        df[postal_col].astype(str) + " " + 
        df[city_col].astype(str) + ", France"
    )
    
    # Géocodage avec barre de progression
    st.markdown("---")
    st.subheader("🌍 Géocodage en cours...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    coordinates = []
    geocoding_success = []
    total_addresses = len(df)
    
    for i, address in enumerate(df['adresse_complete']):
        progress = (i + 1) / total_addresses
        progress_bar.progress(progress)
        status_text.text(f"Géocodage : {i+1}/{total_addresses} - {address[:50]}...")
        
        lat, lon, success = geocode_address(address)
        coordinates.append((lat, lon))
        geocoding_success.append(success)
        
        # Pause pour éviter les limites de l'API
        time.sleep(1)
    
    # Ajout des coordonnées
    df[['lat', 'lon']] = pd.DataFrame(coordinates)
    df['geocoding_success'] = geocoding_success
    
    # Séparation des adresses géocodées/non géocodées
    df_geocoded = df[df['geocoding_success'] == True].dropna(subset=['lat', 'lon']).reset_index(drop=True)
    df_failed = df[df['geocoding_success'] == False].reset_index(drop=True)
    
    st.markdown("---")
    st.subheader("📍 Résultats du géocodage")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Succès", len(df_geocoded))
    with col2:
        st.metric("❌ Échecs", len(df_failed))
    with col3:
        st.metric("📊 Taux de succès", f"{len(df_geocoded)/len(df)*100:.1f}%")
    
    # Affichage des échecs
    if len(df_failed) > 0:
        st.warning("⚠️ Adresses non géocodées :")
        st.dataframe(df_failed[[address_col, postal_col, city_col]], use_container_width=True)
    
    if len(df_geocoded) < 2:
        st.error("❌ Pas assez d'adresses géocodées pour créer une tournée")
        st.stop()
    
    # Détection automatique des points aberrants
    st.markdown("---")
    st.subheader("🔍 Analyse des adresses")
    
    with st.spinner("Détection automatique des points aberrants..."):
        df_main, df_outliers = detect_outliers_automatic(df_geocoded)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Adresses principales", len(df_main))
    with col2:
        st.metric("⚠️ Adresses aberrantes", len(df_outliers))
    
    # Affichage des points aberrants
    if len(df_outliers) > 0:
        st.warning("🚨 Adresses détectées comme aberrantes (trop éloignées du groupe principal) :")
        st.dataframe(df_outliers[[address_col, postal_col, city_col]], use_container_width=True)
        
        # Option pour inclure quand même les aberrants
        include_outliers = st.checkbox("Inclure les adresses aberrantes dans la tournée", value=False)
        if include_outliers:
            df_main = pd.concat([df_main, df_outliers]).reset_index(drop=True)
            st.info("📍 Adresses aberrantes incluses dans la tournée")
    
    if len(df_main) < 2:
        st.error("❌ Pas assez d'adresses valides pour créer une tournée")
        st.stop()
    
    # Optimisation de la tournée
    st.markdown("---")
    st.subheader("🎯 Optimisation de la tournée")
    
    with st.spinner("Calcul de l'itinéraire optimal..."):
        optimal_order = optimize_delivery_route(df_main)
        df_optimized = df_main.loc[optimal_order].reset_index(drop=True)
    
    # Calcul des statistiques
    total_distance, avg_distance, max_distance = calculate_route_stats(df_optimized)
    estimated_time = total_distance * 3  # 3 minutes par km (incluant arrêts)
    
    # Affichage des résultats
    st.markdown("---")
    st.subheader("📊 Résultats de l'optimisation")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏠 Livraisons", len(df_optimized))
    with col2:
        st.metric("📏 Distance totale", f"{total_distance:.1f} km")
    with col3:
        st.metric("⏱️ Temps estimé", f"{estimated_time:.0f} min")
    with col4:
        st.metric("📍 Distance moyenne", f"{avg_distance:.1f} km")
    
    # Tableau optimisé
    st.subheader("🗂️ Itinéraire de livraison optimisé")
    
    # Préparation de l'affichage
    df_display = df_optimized.copy()
    df_display.insert(0, 'Ordre', range(1, len(df_display) + 1))
    
    # Calcul des distances entre étapes
    distances_etapes = []
    for i in range(len(df_display)):
        if i == 0:
            distances_etapes.append("Départ")
        else:
            coord1 = (df_display.iloc[i-1]['lat'], df_display.iloc[i-1]['lon'])
            coord2 = (df_display.iloc[i]['lat'], df_display.iloc[i]['lon'])
            dist = geodesic(coord1, coord2).kilometers
            distances_etapes.append(f"{dist:.1f} km")
    
    df_display['Distance'] = distances_etapes
    
    # Colonnes à afficher
    display_columns = ['Ordre', address_col, postal_col, city_col, 'Distance', 'lat', 'lon']
    
    st.dataframe(
        df_display[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Export Excel
    st.markdown("---")
    st.subheader("💾 Téléchargement")
    
    # Préparation de l'export
    export_data = df_display.copy()
    
    # Fichier Excel avec plusieurs feuilles
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille principale
        export_data.to_excel(writer, index=False, sheet_name='Tournée_optimisée')
        
        # Feuille des échecs de géocodage
        if len(df_failed) > 0:
            df_failed.to_excel(writer, index=False, sheet_name='Échecs_géocodage')
        
        # Feuille des points aberrants
        if len(df_outliers) > 0 and not include_outliers:
            df_outliers.to_excel(writer, index=False, sheet_name='Points_aberrants')
    
    output.seek(0)
    
    # Nom du fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tournee_livraison_{timestamp}.xlsx"
    
    st.download_button(
        label="📥 Télécharger la tournée optimisée",
        data=output,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type="primary",
        use_container_width=True
    )
    
    # Résumé final
    st.success("✅ Tournée optimisée avec succès !")
    st.markdown(f"**🎯 {len(df_optimized)} livraisons** organisées sur **{total_distance:.1f} km** en **{estimated_time:.0f} minutes**")

# Footer
st.markdown("---")
st.markdown("🚛 **Organisateur de Tournées Automatique** - Optimisation intelligente pour vos livraisons")
st.markdown("*Géocodage automatique • Détection d'aberrants • Optimisation de routes • Export Excel*")
