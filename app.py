import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import io
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# --- Page setup ---
st.set_page_config(page_title="Organisateur d'adresses", layout="wide")
st.title("Tournées organisées")

# --- File upload ---
uploaded = st.file_uploader("Chargez un fichier Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()

df = pd.read_excel(uploaded)

# --- Column selection ---
st.sidebar.header("Colonnes des adresses")
cols = df.columns.tolist()
addr_col = st.sidebar.selectbox("Adresse", cols)
pc_col   = st.sidebar.selectbox("Code postal", cols)
city_col = st.sidebar.selectbox("Ville", cols)
if not all([addr_col, pc_col, city_col]):
    st.error("Merci de choisir les 3 colonnes.")
    st.stop()

# --- Build full address and geocode ---
df['address_full'] = df[addr_col].astype(str)+", "+df[pc_col].astype(str)+" "+df[city_col].astype(str)+", France"

geolocator = Nominatim(user_agent="tour_app")
@st.cache_data
def geocode(a):
    try:
        loc = geolocator.geocode(a, timeout=10)
        return (loc.latitude, loc.longitude)
    except:
        return (None, None)

with st.spinner("Géocodage..."):
    df[['lat','lon']] = pd.DataFrame([geocode(a) for a in df['address_full']])
df = df.dropna(subset=['lat','lon']).reset_index(drop=True)

# --- Create distance matrix ---
n = len(df)
dist_matrix = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i==j: continue
        dist_matrix[i][j] = int(geodesic((df.at[i,'lat'],df.at[i,'lon']), (df.at[j,'lat'],df.at[j,'lon'])).meters)

# --- Solve TSP with OR-Tools ---
manager = pywrapcp.RoutingIndexManager(n, 1, 0)
routing = pywrapcp.RoutingModel(manager)

# Distance callback
transit_callback_index = routing.RegisterTransitCallback(
    lambda i,j: dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Setting first node as start
routing.SetStart(0)

# Solve
def init_search():
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.time_limit.FromSeconds(10)
    return routing.SolveWithParameters(search_params)

with st.spinner("Optimisation du parcours..."):
    solution = init_search()
    if not solution:
        st.error("Impossible de résoudre l'itinéraire.")
        st.stop()

# --- Extract route order ---
index = routing.Start(0)
order = []
while not routing.IsEnd(index):
    node = manager.IndexToNode(index)
    order.append(node)
    index = solution.Value(routing.NextVar(index))
order.append(order[0])  # retour optionnel

# --- Display table ---
df_route = df.loc[order[:-1]].reset_index(drop=True)
st.subheader("Ordre des visites")
st.dataframe(df_route[[addr_col, pc_col, city_col]], use_container_width=True)

# --- Export/xlsx ---
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as w:
    df_route.to_excel(w, index=False, sheet_name='Tournée')
out.seek(0)
st.download_button("Télécharger la tournée (.xlsx)", out, file_name='tournee.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
