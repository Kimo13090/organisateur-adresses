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
            time.sleep(2)
    
    return (None, None, False)

def find_center_city_point(df_points):
    """
    Trouve le point central de la ville (zone avec la plus forte densité de livraisons)
    """
    if len(df_points) < 3:
        return df_points.iloc[0].name
    
    # Calcul de la densité pour chaque point
    density_scores = []
    
    for idx, row in df_points.iterrows():
        current_coords = (row['lat'], row['lon'])
        
        # Compter les points dans un rayon de 2km
        nearby_count = 0
        for idx2, row2 in df_points.iterrows():
            if idx != idx2:
                point_coords = (row2['lat'], row2['lon'])
                distance = geodesic(current_coords, point_coords).kilometers
                if distance <= 2.0:  # Rayon de 2km
                    nearby_count += 1
        
        density_scores.append((idx, nearby_count, current_coords))
    
    # Trier par densité décroissante
    density_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retourner l'index du point avec la plus forte densité
    return density_scores[0][0]

def filter_addresses_by_proximity(df_points, max_radius_km=15):
    """
    Filtre les adresses pour garder seulement celles dans un secteur cohérent
    """
    if len(df_points) < 3:
        return df_points, pd.DataFrame()
    
    # Étape 1: Trouver le centre de distribution (zone dense)
    center_idx = find_center_city_point(df_points)
    center_coords = (df_points.at[center_idx, 'lat'], df_points.at[center_idx, 'lon'])
    
    # Étape 2: Calculer les distances depuis le centre
    distances_from_center = []
    for idx, row in df_points.iterrows():
        point_coords = (row['lat'], row['lon'])
        distance = geodesic(center_coords, point_coords).kilometers
        distances_from_center.append((idx, distance))
    
    # Étape 3: Analyse de la distribution des distances
    distances_only = [d[1] for d in distances_from_center]
    
    # Calcul automatique du rayon optimal
    if len(distances_only) > 10:
        # Méthode des quantiles pour détecter les valeurs aberrantes
        q75 = np.percentile(distances_only, 75)
        q25 = np.percentile(distances_only, 25)
        iqr = q75 - q25
        
        # Seuil adaptatif
        max_distance = min(max_radius_km, q75 + 1.5 * iqr)
    else:
        # Pour de petits échantillons, utiliser la médiane + écart-type
        median_dist = np.median(distances_only)
        std_dist = np.std(distances_only)
        max_distance = min(max_radius_km, median_dist + 2 * std_dist)
    
    # Assurer un minimum de 5km pour éviter d'être trop restrictif
    max_distance = max(5.0, max_distance)
    
    # Étape 4: Filtrage des adresses
    addresses_in_sector = []
    addresses_out_of_sector = []
    
    for idx, distance in distances_from_center:
        if distance <= max_distance:
            addresses_in_sector.append(idx)
        else:
            addresses_out_of_sector.append(idx)
    
    df_in_sector = df_points.loc[addresses_in_sector].copy()
    df_out_of_sector = df_points.loc[addresses_out_of_sector].copy()
    
    return df_in_sector, df_out_of_sector

def optimize_delivery_route_from_center(df_points):
    """
    Optimisation de tournée qui commence par le centre-ville
    """
    if len(df_points) <= 1:
        return df_points.index.tolist()
    
    # Étape 1: Identifier le point de départ (centre-ville)
    start_idx = find_center_city_point(df_points)
    
    # Étape 2: Créer des clusters/zones pour organiser la tournée
    route_order = create_smart_route_from_center(df_points, start_idx)
    
    # Étape 3: Optimisation locale avec 2-opt
    route_order = improve_route_2opt(df_points, route_order)
    
    return route_order

def create_smart_route_from_center(df_points, start_idx):
    """
    Crée un itinéraire intelligent qui part du centre et suit une logique géographique
    """
    if len(df_points) <= 2:
        return df_points.index.tolist()
    
    # Point de départ
    start_coords = (df_points.at[start_idx, 'lat'], df_points.at[start_idx, 'lon'])
    
    # Organiser les points par direction/angle depuis le centre
    points_with_angles = []
    
    for idx, row in df_points.iterrows():
        if idx == start_idx:
            continue
            
        point_coords = (row['lat'], row['lon'])
        
        # Calculer l'angle par rapport au centre
        lat_diff = point_coords[0] - start_coords[0]
        lon_diff = point_coords[1] - start_coords[1]
        angle = math.atan2(lat_diff, lon_diff)
        
        # Calculer la distance
        distance = geodesic(start_coords, point_coords).kilometers
        
        points_with_angles.append((idx, angle, distance))
    
    # Trier par angle pour créer un parcours circulaire
    points_with_angles.sort(key=lambda x: x[1])
    
    # Créer l'itinéraire : départ du centre, puis parcours circulaire
    route = [start_idx]
    
    # Ajouter les points dans l'ordre angulaire, en privilégiant les plus proches d'abord
    remaining_points = points_with_angles.copy()
    
    while remaining_points:
        # Trouver le point le plus proche parmi ceux restants
        current_coords = (df_points.at[route[-1], 'lat'], df_points.at[route[-1], 'lon'])
        
        best_idx = None
        best_distance = float('inf')
        best_position = None
        
        for i, (idx, angle, dist_from_center) in enumerate(remaining_points):
            point_coords = (df_points.at[idx, 'lat'], df_points.at[idx, 'lon'])
            distance_to_current = geodesic(current_coords, point_coords).kilometers
            
            # Pondérer par la distance au point actuel et la distance au centre
            # Privilégier les points proches du point actuel
            score = distance_to_current + (dist_from_center * 0.1)
            
            if score < best_distance:
                best_distance = score
                best_idx = idx
                best_position = i
        
        route.append(best_idx)
        remaining_points.pop(best_position)
    
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
    max_iterations = min(50, len(route))  # Réduire les itérations
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Ne pas modifier le premier point (centre-ville)
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

# Interface utilisateur
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

# Paramètres de filtrage
st.markdown("---")
st.subheader("⚙️ Paramètres de la tournée")

col1, col2 = st.columns(2)
with col1:
    max_radius = st.slider(
        "🎯 Rayon maximum de livraison (km)",
        min_value=5,
        max_value=25,
        value=15,
        help="Distance maximale depuis le centre-ville"
    )
with col2:
    max_addresses = st.slider(
        "📦 Nombre maximum d'adresses",
        min_value=10,
        max_value=200,
        value=100,
        help="Limite pour éviter les quotas API"
    )

# Validation et nettoyage des données
st.markdown("---")
st.subheader("🧹 Nettoyage des données")

# Nettoyage des données
df_clean = df.dropna(subset=[address_col, postal_col, city_col]).copy()

# Nettoyage des codes postaux
df_clean[postal_col] = df_clean[postal_col].astype(str).str.extract('(\d+)')[0]
df_clean = df_clean.dropna(subset=[postal_col])

# Suppression des lignes avec des valeurs vides
df_clean = df_clean[df_clean[address_col].astype(str).str.strip() != '']
df_clean = df_clean[df_clean[city_col].astype(str).str.strip() != '']
df_clean = df_clean[df_clean[postal_col].astype(str).str.strip() != '']

# Limitation
if len(df_clean) > max_addresses:
    st.warning(f"⚠️ Limitation à {max_addresses} adresses pour éviter les quotas API")
    df_clean = df_clean.head(max_addresses)

st.info(f"📊 Données nettoyées : {len(df_clean)} adresses valides")

if len(df_clean) == 0:
    st.error("❌ Aucune donnée valide après nettoyage")
    st.stop()

# Bouton de traitement
if st.button("🚀 Organiser la tournée intelligente", type="primary", use_container_width=True):
    
    # Construction des adresses complètes
    df_clean['adresse_complete'] = (
        df_clean[address_col].astype(str) + ", " + 
        df_clean[postal_col].astype(str) + " " + 
        df_clean[city_col].astype(str) + ", France"
    )
    
    # Géocodage avec barre de progression
    st.markdown("---")
    st.subheader("🌍 Géocodage en cours...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    coordinates = []
    geocoding_success = []
    total_addresses = len(df_clean)
    
    for i, address in enumerate(df_clean['adresse_complete']):
        progress = (i + 1) / total_addresses
        progress_bar.progress(progress)
        status_text.text(f"Géocodage : {i+1}/{total_addresses} - {address[:50]}...")
        
        lat, lon, success = geocode_address(address)
        coordinates.append((lat, lon))
        geocoding_success.append(success)
        
        time.sleep(1)
    
    # Ajout des coordonnées
    df_clean[['lat', 'lon']] = pd.DataFrame(coordinates)
    df_clean['geocoding_success'] = geocoding_success
    
    # Séparation des adresses géocodées/non géocodées
    df_geocoded = df_clean[df_clean['geocoding_success'] == True].dropna(subset=['lat', 'lon']).reset_index(drop=True)
    df_failed = df_clean[df_clean['geocoding_success'] == False].reset_index(drop=True)
    
    st.markdown("---")
    st.subheader("📍 Résultats du géocodage")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Succès", len(df_geocoded))
    with col2:
        st.metric("❌ Échecs", len(df_failed))
    with col3:
        st.metric("📊 Taux de succès", f"{len(df_geocoded)/len(df_clean)*100:.1f}%")
    
    if len(df_failed) > 0:
        st.warning("⚠️ Adresses non géocodées :")
        st.dataframe(df_failed[[address_col, postal_col, city_col]], use_container_width=True)
    
    if len(df_geocoded) < 2:
        st.error("❌ Pas assez d'adresses géocodées pour créer une tournée")
        st.stop()
    
    # Filtrage intelligent des adresses
    st.markdown("---")
    st.subheader("🎯 Sélection du secteur de livraison")
    
    with st.spinner("Analyse du secteur et sélection des adresses..."):
        df_sector, df_out_sector = filter_addresses_by_proximity(df_geocoded, max_radius)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Adresses dans le secteur", len(df_sector))
    with col2:
        st.metric("🚫 Adresses hors secteur", len(df_out_sector))
    
    if len(df_out_sector) > 0:
        st.warning("🚨 Adresses exclues (trop éloignées du secteur principal) :")
        st.dataframe(df_out_sector[[address_col, postal_col, city_col]], use_container_width=True)
    
    if len(df_sector) < 2:
        st.error("❌ Pas assez d'adresses dans le secteur pour créer une tournée")
        st.stop()
    
    # Optimisation de la tournée
    st.markdown("---")
    st.subheader("🎯 Optimisation de la tournée")
    
    with st.spinner("Calcul de l'itinéraire optimal depuis le centre-ville..."):
        optimal_order = optimize_delivery_route_from_center(df_sector)
        df_optimized = df_sector.loc[optimal_order].reset_index(drop=True)
    
    # Calcul des statistiques
    total_distance, avg_distance, max_distance = calculate_route_stats(df_optimized)
    estimated_time = total_distance * 3 + len(df_optimized) * 5  # 3 min/km + 5 min/arrêt
    
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
        st.metric("📍 Distance moy/étape", f"{avg_distance:.1f} km")
    
    # Identification du point de départ
    center_idx = find_center_city_point(df_optimized)
    st.info(f"🎯 **Point de départ** : {df_optimized.iloc[0][address_col]} (Centre-ville détecté)")
    
    # Tableau optimisé
    st.subheader("🗂️ Itinéraire de livraison optimisé")
    
    # Préparation de l'affichage
    df_display = df_optimized.copy()
    df_display.insert(0, 'Ordre', range(1, len(df_display) + 1))
    
    # Calcul des distances entre étapes
    distances_etapes = []
    temps_cumule = 0
    
    for i in range(len(df_display)):
        if i == 0:
            distances_etapes.append("🏁 Départ")
        else:
            coord1 = (df_display.iloc[i-1]['lat'], df_display.iloc[i-1]['lon'])
            coord2 = (df_display.iloc[i]['lat'], df_display.iloc[i]['lon'])
            dist = geodesic(coord1, coord2).kilometers
            temps_cumule += dist * 3 + 5  # 3 min/km + 5 min arrêt
            distances_etapes.append(f"{dist:.1f} km")
    
    df_display['Distance'] = distances_etapes
    
    # Colonnes à afficher
    display_columns = ['Ordre', address_col, postal_col, city_col, 'Distance']
    
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
    
    # Ajout des coordonnées pour GPS
    export_data['Latitude'] = df_optimized['lat']
    export_data['Longitude'] = df_optimized['lon']
    
    # Fichier Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille principale
        export_data.to_excel(writer, index=False, sheet_name='Tournée_optimisée')
        
        # Feuille des échecs
        if len(df_failed) > 0:
            df_failed.to_excel(writer, index=False, sheet_name='Échecs_géocodage')
        
        # Feuille des adresses hors secteur
        if len(df_out_sector) > 0:
            df_out_sector.to_excel(writer, index=False, sheet_name='Hors_secteur')
    
    output.seek(0)
    
    # Nom du fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tournee_optimisee_{timestamp}.xlsx"
    
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
    st.markdown(f"**🎯 {len(df_optimized)} livraisons** dans le secteur sur **{total_distance:.1f} km** en **{estimated_time:.0f} minutes**")
    
    # Conseils
    st.markdown("---")
    st.subheader("💡 Conseils d'optimisation")
    st.info("""
    ✅ **Itinéraire optimisé** : Départ du centre-ville vers la périphérie
    ✅ **Secteur cohérent** : Adresses trop éloignées automatiquement exclues
    ✅ **Ordre logique** : Parcours géographiquement cohérent
    ✅ **Temps réaliste** : Inclut temps de conduite + temps d'arrêt
    """)

# Footer
st.markdown("---")
st.markdown("🚛 **Organisateur de Tournées Automatique** - Optimisation intelligente pour vos livraisons")
st.markdown("*Géocodage automatique • Filtrage intelligent • Optimisation depuis centre-ville • Export Excel*")
