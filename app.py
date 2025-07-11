import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
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

# --- Géocodage des adresses avec cache ---
geolocator = Nominatim(user_agent="streamlit_app")
@st.cache_data
def geocode(addr):
    try:
        loc = geolocator.geocode(addr, timeout=10)
        return (loc.latitude, loc.longitude)
    except:
        return (None, None)

with st.spinner("Géocodage en cours..."):
    df[['lat','lon']] = pd.DataFrame([geocode(a) for a in df['full_address']])

# Filtrer adresses géocodées
df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
if df.empty:
    st.error("Aucune adresse valide n'a pu être géocodée.")
    st.stop()
st.success(f"{len(df)} adresses géocodées.")

# --- Filtrage par clustering DBSCAN pour garder le cluster principal ---
coords = df[['lat','lon']].to_numpy()
# eps en degrés ≈ 0.02 deg ~ 2.2km
db = DBSCAN(eps=0.02, min_samples=2).fit(coords)
labels = db.labels_
df['cluster'] = labels
# Choisir le cluster le plus grand (hors -1 noise)
cluster_sizes = df[df['cluster']!=-1]['cluster'].value_counts()
if cluster_sizes.empty:
    st.error("Aucun cluster principal trouvé. Toutes adresses sont isolées.")
    st.stop()
main_cluster = int(cluster_sizes.idxmax())
df_in = df[df['cluster']==main_cluster].reset_index(drop=True)
df_out = df[df['cluster']!=main_cluster].reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur principal :")
    st.dataframe(df_out[[addr_col, pc_col, city_col]])

# --- Tri glouton (nearest neighbor) ---
def greedy_order(df_pts, start_idx):
    visited = [start_idx]
    remaining = set(range(len(df_pts))) - {start_idx}
    while remaining:
        last = visited[-1]
        # calculer distances aux non-visités
        dists = {i: geodesic(
            (df_pts.at[last,'lat'], df_pts.at[last,'lon']),
            (df_pts.at[i,'lat'], df_pts.at[i,'lon'])
        ).meters for i in remaining}
        nxt = min(dists, key=dists.get)
        visited.append(nxt)
        remaining.remove(nxt)
    return visited

# Déterminer index de départ : adresse la plus centrale (min somme distances)
sum_dists = df_in.apply(
    lambda r: df_in.apply(
        lambda x: geodesic((r['lat'],r['lon']), (x['lat'],x['lon'])).meters,
        axis=1
    ).sum(), axis=1
)
start_idx = int(sum_dists.idxmin())
order = greedy_order(df_in, start_idx)
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
