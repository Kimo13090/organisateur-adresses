import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
import time
import numpy as np
from datetime import datetime
import math

# Configuration de la page
st.set_page_config(
    page_title="Organisateur de Tournées Excel",
    page_icon="🚛",
    layout="wide"
)

# CSS simplifié
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
    }
    
    .success-box {
        background: #48bb78;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .stats-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .footer {
        background: #2d3748;
        color: white;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="main-header">
    <h1>🚛 Organisateur de Tournées Excel</h1>
    <p>Transformez vos adresses en itinéraires optimisés</p>
</div>
""", unsafe_allow_html=True)

# Cache pour le géocodage
if 'geocode_cache' not in st.session_state:
    st.session_state.geocode_cache = {}

@st.cache_data(ttl=3600)
def geocode_address(address):
    """Géocode une adresse avec cache"""
    if address in st.session_state.geocode_cache:
        return st.session_state.geocode_cache[address]
    
    geolocator = Nominatim(user_agent="tournee_optimizer_v4", timeout=15)
    
    try:
        # Essayer plusieurs fois si nécessaire
        for attempt in range(3):
            location = geolocator.geocode(address, country_codes=['fr'])
            if location:
                result = (location.latitude, location.longitude, True)
                st.session_state.geocode_cache[address] = result
                return result
            time.sleep(1)
        
        # Si pas trouvé après 3 essais
        result = (None, None, False)
        st.session_state.geocode_cache[address] = result
        return result
        
    except Exception as e:
        result = (None, None, False)
        st.session_state.geocode_cache[address] = result
        return result

def find_city_center(df_points):
    """Trouve le centre-ville (barycentre des points)"""
    center_lat = df_points['lat'].mean()
    center_lon = df_points['lon'].mean()
    
    # Trouver le point le plus proche du barycentre
    min_dist = float('inf')
    center_idx = df_points.index[0]
    
    for idx, row in df_points.iterrows():
        dist = geodesic((center_lat, center_lon), (row['lat'], row['lon'])).kilometers
        if dist < min_dist:
            min_dist = dist
            center_idx = idx
    
    return center_idx

def optimize_route_from_center(df_points):
    """
    Crée un itinéraire optimisé en partant du centre-ville
    Utilise l'algorithme du plus proche voisin amélioré
    """
    if len(df_points) <= 1:
        return df_points.index.tolist()
    
    # Trouver le point de départ (centre-ville)
    start_idx = find_city_center(df_points)
    
    # Algorithme du plus proche voisin
    route = [start_idx]
    unvisited = set(df_points.index) - {start_idx}
    current_idx = start_idx
    
    while unvisited:
        current_point = (df_points.at[current_idx, 'lat'], df_points.at[current_idx, 'lon'])
        
        # Trouver le point non visité le plus proche
        nearest_idx = None
        nearest_dist = float('inf')
        
        for idx in unvisited:
            point = (df_points.at[idx, 'lat'], df_points.at[idx, 'lon'])
            dist = geodesic(current_point, point).kilometers
            
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        
        # Ajouter à la route
        route.append(nearest_idx)
        unvisited.remove(nearest_idx)
        current_idx = nearest_idx
    
    return route

def improve_route_2opt(df_points, route):
    """Améliore la route avec l'algorithme 2-opt"""
    if len(route) < 4:
        return route
    
    improved = True
    best_route = route.copy()
    
    while improved:
        improved = False
        
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                
                # Créer une nouvelle route en inversant le segment
                new_route = best_route.copy()
                new_route[i:j] = best_route[i:j][::-1]
                
                # Calculer les distances
                old_dist = 0
                new_dist = 0
                
                # Distance de l'ancienne route pour les segments affectés
                if i > 0:
                    old_dist += geodesic(
                        (df_points.at[best_route[i-1], 'lat'], df_points.at[best_route[i-1], 'lon']),
                        (df_points.at[best_route[i], 'lat'], df_points.at[best_route[i], 'lon'])
                    ).kilometers
                    new_dist += geodesic(
                        (df_points.at[new_route[i-1], 'lat'], df_points.at[new_route[i-1], 'lon']),
                        (df_points.at[new_route[i], 'lat'], df_points.at[new_route[i], 'lon'])
                    ).kilometers
                
                if j < len(route):
                    old_dist += geodesic(
                        (df_points.at[best_route[j-1], 'lat'], df_points.at[best_route[j-1], 'lon']),
                        (df_points.at[best_route[j], 'lat'], df_points.at[best_route[j], 'lon'])
                    ).kilometers if j < len(route) else 0
                    new_dist += geodesic(
                        (df_points.at[new_route[j-1], 'lat'], df_points.at[new_route[j-1], 'lon']),
                        (df_points.at[new_route[j], 'lat'], df_points.at[new_route[j], 'lon'])
                    ).kilometers if j < len(route) else 0
                
                # Si amélioration, garder la nouvelle route
                if new_dist < old_dist:
                    best_route = new_route
                    improved = True
                    break
            
            if improved:
                break
    
    return best_route

def split_into_multiple_tours(df_points, max_per_tour=50):
    """Divise en plusieurs tournées si nécessaire"""
    if len(df_points) <= max_per_tour:
        # Une seule tournée
        route = optimize_route_from_center(df_points)
        route = improve_route_2opt(df_points, route)
        return [df_points.loc[route]]
    
    # Plusieurs tournées nécessaires
    tours = []
    remaining = df_points.copy()
    
    while len(remaining) > 0:
        # Prendre jusqu'à max_per_tour points
        if len(remaining) <= max_per_tour:
            # Dernière tournée
            route = optimize_route_from_center(remaining)
            route = improve_route_2opt(remaining, route)
            tours.append(remaining.loc[route])
            break
        else:
            # Trouver le centre de ce qui reste
            center_idx = find_city_center(remaining)
            center_point = (remaining.at[center_idx, 'lat'], remaining.at[center_idx, 'lon'])
            
            # Calculer les distances depuis le centre
            distances = []
            for idx, row in remaining.iterrows():
                point = (row['lat'], row['lon'])
                dist = geodesic(center_point, point).kilometers
                distances.append((idx, dist))
            
            # Trier par distance et prendre les plus proches
            distances.sort(key=lambda x: x[1])
            tour_indices = [d[0] for d in distances[:max_per_tour]]
            
            # Créer la tournée
            tour_df = remaining.loc[tour_indices]
            route = optimize_route_from_center(tour_df)
            route = improve_route_2opt(tour_df, route)
            tours.append(tour_df.loc[route])
            
            # Retirer de remaining
            remaining = remaining.drop(tour_indices)
    
    return tours

def calculate_distances(df_route):
    """Calcule les distances entre chaque point"""
    distances = []
    total_distance = 0
    
    for i in range(len(df_route)):
        if i == 0:
            distances.append(0)
        else:
            coord1 = (df_route.iloc[i-1]['lat'], df_route.iloc[i-1]['lon'])
            coord2 = (df_route.iloc[i]['lat'], df_route.iloc[i]['lon'])
            dist = geodesic(coord1, coord2).kilometers
            distances.append(dist)
            total_distance += dist
    
    df_route['distance_km'] = distances
    return df_route, total_distance

def detect_columns(df):
    """Détection automatique des colonnes"""
    columns = df.columns.tolist()
    
    # Patterns de détection
    address_patterns = ['adresse', 'address', 'rue', 'street', 'client', 'nom']
    postal_patterns = ['postal', 'cp', 'code', 'zip']
    city_patterns = ['ville', 'city', 'commune']
    
    address_col = None
    postal_col = None
    city_col = None
    
    for col in columns:
        col_lower = col.lower()
        
        if not address_col and any(p in col_lower for p in address_patterns):
            address_col = col
        if not postal_col and any(p in col_lower for p in postal_patterns):
            postal_col = col
        if not city_col and any(p in col_lower for p in city_patterns):
            city_col = col
    
    return address_col, postal_col, city_col

def generate_excel(tours, df_failed, address_col, postal_col, city_col):
    """Génère le fichier Excel avec toutes les tournées"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Résumé global
        summary = []
        total_distance_global = 0
        total_addresses = 0
        
        # Une feuille par tournée
        for idx, tour in enumerate(tours):
            sheet_name = f'Tournée_{idx + 1}'
            
            # Préparer les données
            export = tour.copy()
            export.insert(0, 'Ordre', range(1, len(export) + 1))
            export.insert(1, 'Tournée', idx + 1)
            
            # Distance et temps
            _, tour_distance = calculate_distances(export)
            total_distance_global += tour_distance
            total_addresses += len(export)
            
            # Temps estimé (3 min/km + 5 min/arrêt)
            temps_trajet = tour_distance * 3
            temps_arrets = len(export) * 5
            temps_total = temps_trajet + temps_arrets
            
            # Heures estimées
            heure_depart = pd.Timestamp('08:00:00')
            export['Heure_Estimée'] = [
                (heure_depart + pd.Timedelta(minutes=i*5 + sum(export['distance_km'][:i])*3)).strftime('%H:%M')
                for i in range(len(export))
            ]
            
            # Colonnes à exporter
            cols = ['Ordre', 'Tournée', address_col, postal_col, city_col, 'distance_km', 'Heure_Estimée', 'lat', 'lon']
            export[cols].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Ajouter au résumé
            summary.append({
                'Tournée': idx + 1,
                'Adresses': len(export),
                'Distance (km)': round(tour_distance, 1),
                'Temps estimé': f"{int(temps_total//60)}h{int(temps_total%60)}min",
                'Départ': '08:00',
                'Arrivée': (heure_depart + pd.Timedelta(minutes=temps_total)).strftime('%H:%M')
            })
        
        # Feuille résumé
        pd.DataFrame(summary).to_excel(writer, sheet_name='Résumé', index=False)
        
        # Statistiques globales
        stats = pd.DataFrame([{
            'Total tournées': len(tours),
            'Total adresses': total_addresses,
            'Total distance (km)': round(total_distance_global, 1),
            'Temps total estimé': f"{int((total_distance_global*3 + total_addresses*5)//60)}h",
            'Adresses non géolocalisées': len(df_failed)
        }])
        stats.to_excel(writer, sheet_name='Statistiques', index=False)
        
        # Échecs de géolocalisation
        if len(df_failed) > 0:
            df_failed[[address_col, postal_col, city_col]].to_excel(
                writer, sheet_name='Non_Géolocalisées', index=False
            )
    
    output.seek(0)
    return output

# Interface principale
st.markdown("### 📁 Importez votre fichier Excel")

uploaded_file = st.file_uploader(
    "Déposez votre fichier avec les adresses à organiser",
    type=["xlsx", "xls"],
    help="Le fichier doit contenir : Adresse, Code Postal, Ville"
)

if uploaded_file:
    # Lecture du fichier
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ **{len(df)} adresses** chargées")
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.stop()
    
    # Détection des colonnes
    address_col, postal_col, city_col = detect_columns(df)
    
    # Sélection manuelle si nécessaire
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
    if st.button("🚀 **Organiser la tournée**", type="primary", use_container_width=True):
        # Préparation des données
        df_clean = df.copy()
        
        # Nettoyage basique
        df_clean = df_clean.dropna(subset=[address_col, postal_col, city_col])
        df_clean[postal_col] = df_clean[postal_col].astype(str).str.strip()
        df_clean[address_col] = df_clean[address_col].astype(str).str.strip()
        df_clean[city_col] = df_clean[city_col].astype(str).str.strip()
        
        # Construction des adresses complètes
        df_clean['adresse_complete'] = (
            df_clean[address_col] + ", " +
            df_clean[postal_col] + " " +
            df_clean[city_col] + ", France"
        )
        
        st.info(f"🔍 Traitement de **{len(df_clean)} adresses**...")
        
        # Géolocalisation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, address in enumerate(df_clean['adresse_complete']):
            # Mise à jour du progrès
            progress = (i + 1) / len(df_clean)
            progress_bar.progress(progress)
            status_text.text(f"Géolocalisation : {i+1}/{len(df_clean)}")
            
            # Géocoder
            lat, lon, success = geocode_address(address)
            results.append((lat, lon, success))
            
            # Pause pour respecter les limites API
            if i < len(df_clean) - 1:
                time.sleep(1.0)
        
        # Ajouter les résultats
        df_clean[['lat', 'lon', 'geocoded']] = pd.DataFrame(results)
        
        # Séparer succès/échecs
        df_success = df_clean[df_clean['geocoded'] == True].copy()
        df_failed = df_clean[df_clean['geocoded'] == False].copy()
        
        # Afficher les statistiques
        st.markdown("### 📊 Résultats de la géolocalisation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Géolocalisées", len(df_success))
        with col2:
            st.metric("❌ Non trouvées", len(df_failed))
        with col3:
            st.metric("📊 Taux de succès", f"{len(df_success)/len(df_clean)*100:.1f}%")
        
        if len(df_success) == 0:
            st.error("❌ Aucune adresse n'a pu être géolocalisée")
            st.stop()
        
        # Organisation en tournées
        st.info("🚛 Organisation des tournées...")
        
        # Créer les tournées (max 50 adresses par tournée)
        tours = split_into_multiple_tours(df_success, max_per_tour=50)
        
        # Générer le fichier Excel
        excel_file = generate_excel(tours, df_failed, address_col, postal_col, city_col)
        
        # Zone de succès
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Tournées organisées avec succès !")
        st.markdown(f"**{len(tours)} tournée(s)** créée(s) pour **{len(df_success)} adresses**")
        
        # Bouton de téléchargement
        st.download_button(
            label="📥 **Télécharger le fichier Excel organisé**",
            data=excel_file,
            file_name=f"tournees_organisees_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Détails des tournées
        st.markdown("### 🗺️ Détail des tournées")
        
        for idx, tour in enumerate(tours):
            tour_with_dist, total_dist = calculate_distances(tour)
            temps_total = total_dist * 3 + len(tour) * 5
            
            with st.expander(f"📍 Tournée {idx + 1} - {len(tour)} adresses"):
                # Statistiques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📦 Adresses", len(tour))
                with col2:
                    st.metric("📏 Distance", f"{total_dist:.1f} km")
                with col3:
                    st.metric("⏱️ Durée", f"{int(temps_total//60)}h{int(temps_total%60)}min")
                
                # Afficher les premières adresses
                display_df = tour[[address_col, postal_col, city_col]].copy()
                display_df.insert(0, 'Ordre', range(1, len(display_df) + 1))
                st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
                
                if len(tour) > 10:
                    st.info(f"... et {len(tour) - 10} autres adresses")
        
        # Adresses non géolocalisées
        if len(df_failed) > 0:
            with st.expander(f"⚠️ {len(df_failed)} adresses non géolocalisées"):
                st.dataframe(
                    df_failed[[address_col, postal_col, city_col]],
                    use_container_width=True,
                    hide_index=True
                )

else:
    # Instructions
    st.info("""
    ### 📋 Mode d'emploi :
    
    1. **Préparez** votre fichier Excel avec les colonnes : Adresse, Code Postal, Ville
    2. **Importez** le fichier (jusqu'à 250 adresses)
    3. **Vérifiez** que les colonnes sont bien détectées
    4. **Cliquez** sur "Organiser la tournée"
    5. **Téléchargez** le fichier Excel avec les tournées optimisées
    
    ✅ **Toutes les adresses seront traitées** et organisées en tournées logiques
    ✅ **Départ du centre-ville** automatiquement détecté
    ✅ **Ordre optimisé** pour minimiser les distances
    """)

# Footer
st.markdown("""
<div class="footer">
    <p style="margin: 0; font-size: 1.1rem;">🚛 Organisateur de Tournées Excel</p>
    <p style="margin: 0.5rem 0; opacity: 0.9;">Solution 100% gratuite • Traitement complet • Export Excel optimisé</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Par l'alternant Delestret Kim</p>
</div>
""", unsafe_allow_html=True)
