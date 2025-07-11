import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io

# --- Configuration de la page ---
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Tournées organisées (sans carte)")

# --- Téléversement du fichier Excel ---
uploaded = st.file_uploader("Chargez un fichier Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()
try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Erreur lecture du fichier Excel : {e}")
    st.stop()

# --- Sélection des colonnes ---
st.sidebar.header("Paramètres des colonnes")
cols = df.columns.tolist()
addr_col = st.sidebar.selectbox("Colonne adresse", cols)
pc_col   = st.sidebar.selectbox("Colonne code postal", cols)
city_col = st.sidebar.selectbox("Colonne ville", cols)
if not all([addr_col, pc_col, city_col]):
    st.error("Merci de sélectionner les 3 colonnes d'adresses.")
    st.stop()

# --- Construction de l'adresse complète ---
df['full_address'] = (
    df[addr_col].astype(str) + ", " +
    df[pc_col].astype(str) + " " +
    df[city_col].astype(str) + ", France"
)

# --- Géocodage des adresses ---
geolocator = Nominatim(user_agent="streamlit_app")
@st.cache_data
# Mise en cache pour accélérer
def geocode(addr):
    try:
        loc = geolocator.geocode(addr, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

with st.spinner("Géocodage en cours..."):
    coords = [geocode(a) for a in df['full_address']]
df['lat'], df['lon'] = zip(*coords)
df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
if df.empty:
    st.error("Aucune adresse n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df)} adresses géocodées.")

# --- Filtrage des adresses hors secteur (>5km du centroïde) ---
centroid = (df['lat'].mean(), df['lon'].mean())
df['dist_to_center'] = df.apply(
    lambda row: geodesic((row['lat'], row['lon']), centroid).meters,
    axis=1
)
df_in = df[df['dist_to_center'] <= 5000].reset_index(drop=True)
df_out = df[df['dist_to_center'] > 5000].reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur (>5km du centre) :")
    st.dataframe(df_out[[addr_col, pc_col, city_col, 'dist_to_center']])
if df_in.empty:
    st.error("Aucune adresse dans le secteur.")
    st.stop()

# --- Fonction de tri glouton (nearest neighbor) ---
def nearest_neighbor_order(df_pts, start_idx=0):
    visited = [start_idx]
    remaining = set(range(len(df_pts))) - {start_idx}
    while remaining:
        last = visited[-1]
        # calcul distances aux non-visités
        dists = {
            i: geodesic(
                (df_pts.at[last,'lat'], df_pts.at[last,'lon']),
                (df_pts.at[i,'lat'], df_pts.at[i,'lon'])
            ).meters
            for i in remaining
        }
        next_idx = min(dists, key=dists.get)
        visited.append(next_idx)
        remaining.remove(next_idx)
    return visited

# --- Détermination du point de départ ---
# par défaut, on prend l'adresse la plus centrale (distance minimale au centroïde)
df_in['sum_distances'] = df_in.apply(
    lambda row: df_in.apply(
        lambda r: geodesic((row['lat'],row['lon']), (r['lat'],r['lon'])).meters,
        axis=1
    ).sum(),
    axis=1
)
start_index = int(df_in['sum_distances'].idxmin())

# --- Calcul de l'ordre de tournée ---
order = nearest_neighbor_order(df_in, start_index)
df_route = df_in.loc[order].reset_index(drop=True)

# --- Affichage du tableau optimisé ---
st.subheader("Ordre des visites optimisé")
st.dataframe(
    df_route[[addr_col, pc_col, city_col, 'lat', 'lon']],
    use_container_width=True
)

# --- Export du résultat en Excel ---
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    df_route.to_excel(writer, index=False, sheet_name='Tournée')
out.seek(0)
st.download_button(
    label="Télécharger la tournée (.xlsx)",
    data=out,
    file_name='tournee_organisee.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
