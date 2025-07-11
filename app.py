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
[df code continues...]
uploaded = st.file_uploader("Chargez un fichier Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()
try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Erreur lecture du fichier Excel : {e}")
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
def geocode(addr):
    try:
        loc = geolocator.geocode(addr, timeout=10)
        return (loc.latitude, loc.longitude)
    except:
        return (None, None)

with st.spinner("Géocodage en cours..."):
    coords = [geocode(a) for a in df['full_address']]
    df[['lat','lon']] = pd.DataFrame(coords)

# Filtrer adresses géocodées
df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
if df.empty:
    st.error("Aucune adresse valide n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df)} adresses géocodées.")

# --- Filtrage automatique des adresses hors secteur via distance au centroïde ---
# Calcul du centroïde
centroid_lat = df['lat'].mean()
centroid_lon = df['lon'].mean()
# Calcul des distances au centroïde
df['dist_centroid'] = df.apply(
    lambda r: geodesic((r['lat'], r['lon']), (centroid_lat, centroid_lon)).meters,
    axis=1
)
# Seuil dynamique : moyenne + 1.5 * écart-type
dist_mean = df['dist_centroid'].mean()
dist_std  = df['dist_centroid'].std()
threshold = dist_mean + 1.5 * dist_std
# Sélection du secteur principal
df_in = df[df['dist_centroid'] <= threshold].reset_index(drop=True)
df_out = df[df['dist_centroid'] > threshold].reset_index(drop=True)
if not df_out.empty:
    st.warning(f"Adresses hors secteur (> {threshold:.0f} m) :")
    st.dataframe(df_out[[addr_col, pc_col, city_col, 'dist_centroid']])
if df_in.empty:
    st.error("Aucune adresse dans le secteur principal après filtrage.")
    st.stop()

# --- Tri glouton (Nearest Neighbor) sur le secteur principal --- (Nearest Neighbor) ---
def greedy_order(df_pts, start_idx):
    visited = [start_idx]
    rem = set(range(len(df_pts))) - {start_idx}
    while rem:
        last = visited[-1]
        # calcul des distances
        dists = {i: geodesic((df_pts.at[last,'lat'], df_pts.at[last,'lon']),
                              (df_pts.at[i,'lat'], df_pts.at[i,'lon'])).meters
                 for i in rem}
        nxt = min(dists, key=dists.get)
        visited.append(nxt)
        rem.remove(nxt)
    return visited

# Déterminer point de départ : adresse la plus centrale (min sum dist)
sum_dists = df_in.apply(
    lambda r: df_in.apply(
        lambda x: geodesic((r['lat'],r['lon']), (x['lat'],x['lon'])).meters,
        axis=1
    ).sum(), axis=1
)
start_index = int(sum_dists.idxmin())
order = greedy_order(df_in, start_index)
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
