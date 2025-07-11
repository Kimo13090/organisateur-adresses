import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
# --- Filtrage par clustering to isolate main cluster sans sklearn ---
# On regroupe les adresses proches (<2000m) en clusters par union-find
coords = df[['lat','lon']].to_numpy()
threshold = 2000  # mètres
# Union-Find
parent = list(range(len(df)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(i,j):
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[rj] = ri

# Construire les liaisons
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if geodesic((coords[i][0], coords[i][1]), (coords[j][0], coords[j][1])).meters <= threshold:
            union(i, j)
# Regrouper par racine
groups = {}
for i in range(len(df)):
    root = find(i)
    groups.setdefault(root, []).append(i)
# Garder le plus grand groupe
groups = {r: idxs for r, idxs in groups.items() if r != -1}
main = max(groups.items(), key=lambda kv: len(kv[1]))[1]
# Sélectionner df_in et df_out
df_in = df.loc[main].reset_index(drop=True)
df_out = df.drop(main).reset_index(drop=True)
if not df_out.empty:
    st.warning("Adresses hors secteur principal :")
    st.dataframe(df_out[[addr_col, pc_col, city_col]])
# --- Tri glouton (nearest neighbor) --- (nearest neighbor) ---
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
