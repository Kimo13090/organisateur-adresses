# --- Import Libraries ---
import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import io
import numpy as np
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Optimisateur de Tournées Logistiques",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS & UI ---
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .header { background: linear-gradient(135deg, #667eea, #764ba2); padding:20px; border-radius:10px; text-align:center; color:#fff; }
    div[data-testid="metric-container"] { background:#fff; border:1px solid #e0e0e0; padding:1rem; border-radius:8px; }
    .stButton>button { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:0.75rem 2rem; border-radius:8px; }
    .download { background:linear-gradient(135deg,#48bb78,#38a169); color:#fff; padding:1.5rem; border-radius:10px; text-align:center; }
    .footer { background:#2d3748; color:#fff; padding:1rem; text-align:center; margin-top:2rem; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🚛 Optimisateur de Tournées Logistiques</h1>
    <p>Optimisez automatiquement vos itinéraires jusqu'à 250 adresses (gratuite & rapide)</p>
</div>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
def init_state():
    for key in [
        'processed', 'df_opt', 'df_out', 'df_fail', 'total_dist',
        'excel_buf', 'map_obj', 'last_uploaded_name']:
        if key not in st.session_state:
            st.session_state[key] = None
init_state()

# --- Smart Column Detector ---
def smart_detect_cols(df):
    cols = df.columns.str.lower()
    addr = next((c for c in df.columns if any(x in c.lower() for x in ['adresse', 'rue', 'address', 'voie'])), None)
    postal = next((c for c in df.columns if any(x in c.lower() for x in ['cp', 'zip', 'postal', 'code'])), None)
    city = next((c for c in df.columns if any(x in c.lower() for x in ['ville', 'city', 'commune'])), None)
    return addr, postal, city

# --- Geocoding with Caching and RateLimiter ---
@st.cache_data(show_spinner=False)
def get_geocoder():
    geolocator = Nominatim(user_agent="logistics_app_v3")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)
    return geocode

def batch_geocode(addresses, limit=250):
    geocode = get_geocoder()
    latlons = []
    for i, adr in enumerate(addresses):
        if i >= limit: break
        loc = geocode(adr)
        latlons.append((loc.latitude, loc.longitude) if loc else (np.nan, np.nan))
    return pd.DataFrame(latlons, columns=['lat','lon'])

# --- Outlier Detection ---
def detect_outliers(df_geo):
    if len(df_geo) < 5:
        return [], df_geo.index.tolist()
    center = (df_geo['lat'].median(), df_geo['lon'].median())
    df_geo['dist_center'] = df_geo.apply(lambda r: geodesic(center,(r.lat,r.lon)).km, axis=1)
    med, mad = np.median(df_geo['dist_center']), np.median(np.abs(df_geo['dist_center']-np.median(df_geo['dist_center'])))
    thresh = max(med + 2.5*mad, 10)
    in_idx = df_geo[df_geo['dist_center']<=thresh].index.tolist()
    out_idx = df_geo[df_geo['dist_center']>thresh].index.tolist()
    return in_idx, out_idx

# --- Optimized Route (NN + 2-opt) ---
def optimize_route(df_pts):
    order = []
    pts = df_pts.copy().reset_index()
    center = pts[['lat','lon']].mean().values.tolist()
    start = pts.assign(dist=lambda d: d.apply(lambda r: geodesic(center,(r.lat,r.lon)).km, axis=1)).nsmallest(1,'dist').index[0]
    current = start
    unvis = set(pts.index) - {current}
    order.append(current)
    while unvis:
        next_idx = min(unvis, key=lambda i: geodesic((pts.loc[current,'lat'],pts.loc[current,'lon']),
                                                    (pts.loc[i,'lat'],pts.loc[i,'lon'])).km)
        order.append(next_idx); unvis.remove(next_idx); current = next_idx
    # 2-opt (local opt.)
    best = order
    improved = True
    def tour_length(ordr):
        return sum(geodesic((pts.loc[a,'lat'],pts.loc[a,'lon']), (pts.loc[b,'lat'],pts.loc[b,'lon'])).km
                   for a,b in zip(ordr, ordr[1:]))
    best_len = tour_length(best)
    while improved:
        improved=False
        for i in range(1,len(best)-2):
            for j in range(i+1,len(best)):
                new = best[:i]+best[i:j+1][::-1]+best[j+1:]
                nl = tour_length(new)
                if nl < best_len:
                    best, best_len, improved = new, nl, True
    df_ord = pts.loc[best].reset_index(drop=True)
    return df_ord, best_len

# --- Distance Columns ---
def add_distances(df_rt):
    dists=[0]
    for i in range(1,len(df_rt)):
        dists.append(geodesic((df_rt.loc[i-1,'lat'],df_rt.loc[i-1,'lon']), (df_rt.loc[i,'lat'],df_rt.loc[i,'lon'])).km)
    df_rt['dist_step']=dists; df_rt['cum_dist']=df_rt['dist_step'].cumsum()
    return df_rt

# --- Folium Map ---
def build_map(df_rt, df_out):
    m=folium.Map(location=[df_rt.lat.mean(),df_rt.lon.mean()], zoom_start=12)
    for idx,row in df_rt.iterrows():
        icon = 'play' if idx==0 else 'stop' if idx==len(df_rt)-1 else 'location-dot'
        clr = 'green' if idx==0 else 'red' if idx==len(df_rt)-1 else 'blue'
        folium.Marker([row.lat,row.lon],
                      popup=f"<b>{idx+1}</b> {row.address if 'address' in row else ''}<br>{row.dist_step:.1f} km",
                      icon=folium.Icon(color=clr,icon=icon,prefix='fa')).add_to(m)
    folium.PolyLine(df_rt[['lat','lon']].values.tolist(), color='purple', weight=3).add_to(m)
    if not df_out.empty:
        for _,r in df_out.iterrows():
            folium.Marker([r.lat,r.lon],popup=r.address if 'address' in r else '', icon=folium.Icon(color='orange',icon='exclamation-triangle',prefix='fa')).add_to(m)
    return m

# --- Excel Export ---
def make_excel(df_rt, df_out, df_fail, tot_dist):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df_rt.assign(Order=range(1,len(df_rt)+1)).to_excel(w,index=False,sheet_name='Itineraire')
        if not df_out.empty: df_out.to_excel(w,index=False,sheet_name='Hors_Secteur')
        if not df_fail.empty: df_fail.to_excel(w,index=False,sheet_name='Echecs')
        synth = pd.DataFrame({
            'Métrique':["Livraisons optimisées", "Hors secteur", "Échecs geocoding", "Distance totale (km)"],
            'Valeur':[len(df_rt), len(df_out), len(df_fail), f"{tot_dist:.2f}"]
        })
        synth.to_excel(w, index=False, sheet_name="Synthese")
    buf.seek(0); return buf

# --- Upload Section ---
st.markdown("### 📁 Importez votre fichier (max 250 adresses)")
upl = st.file_uploader("Fichier Excel", type=['xls','xlsx'])

# --- Main Processing ---
if upl:
    if st.session_state.last_uploaded_name != upl.name:
        st.session_state.processed = None
        st.session_state.last_uploaded_name = upl.name
    df = pd.read_excel(upl)
    st.success(f"{len(df)} lignes importées")
    addr,postal,city = smart_detect_cols(df)
    if not all([addr,postal,city]):
        addr=st.selectbox('Adresse',df.columns)
        postal=st.selectbox('Code Postal',df.columns)
        city=st.selectbox('Ville',df.columns)
    if st.button("🚀 Générer l'itinéraire optimisé"):
        df_clean = df[[addr,postal,city]].dropna().rename(columns={addr:'street',postal:'zip',city:'city'})
        df_clean['address']=df_clean.street.astype(str)+', '+df_clean.zip.astype(str)+' '+df_clean.city+', France'
        # Limiter à 250
        if len(df_clean)>250:
            st.warning("⚠️ Limite gratuite de 250 adresses, seules les 250 premières seront traitées.")
            df_clean = df_clean.iloc[:250]
        # Geocode
        st.info("🌍 Géocodage en cours...")
        df_geo = df_clean.copy()
        coords = batch_geocode(df_geo['address'])
        df_geo = df_geo.join(coords)
        df_fail = df_geo[df_geo[['lat','lon']].isna().any(axis=1)].copy()
        df_geo = df_geo.dropna(subset=['lat','lon']).copy().reset_index(drop=True)
        if len(df_geo)<2:
            st.error("❌ Pas assez d'adresses géolocalisées."); st.stop()
        # Outliers
        st.info("🔎 Détection hors secteur...")
        in_idx, out_idx = detect_outliers(df_geo)
        df_in, df_out = df_geo.loc[in_idx], df_geo.loc[out_idx]
        # Optimisation
        st.info("⚡ Optimisation de l'itinéraire...")
        df_route, total = optimize_route(df_in)
        df_route = add_distances(df_route)
        # Map & Excel
        m = build_map(df_route, df_out)
        buf = make_excel(df_route, df_out, df_fail, total)
        # Store
        st.session_state.processed=True
        st.session_state.df_opt=df_route
        st.session_state.df_out=df_out
        st.session_state.df_fail=df_fail
        st.session_state.total_dist=total
        st.session_state.map_obj=m
        st.session_state.excel_buf=buf
        st.experimental_rerun()

# --- Results Display ---
if st.session_state.processed:
    st.markdown("---")
    st.markdown('<div class="download"><h3>📥 Votre itinéraire est prêt</h3></div>', unsafe_allow_html=True)
    st.download_button('Télécharger Excel', st.session_state.excel_buf,
                      file_name=f"tournee_{datetime.now():%Y%m%d_%H%M}.xlsx")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Livraisons', len(st.session_state.df_opt))
    c2.metric('Distance totale', f"{st.session_state.total_dist:.1f} km")
    eta = int(st.session_state.total_dist*3 + len(st.session_state.df_opt)*5)
    c3.metric('Temps estimé', f"{eta//60}h {eta%60}m")
    c4.metric('Hors secteur', len(st.session_state.df_out))
    st.markdown("### 🗺️ Carte de la tournée")
    st_folium(st.session_state.map_obj, width=800)
    st.markdown("### 📋 Détail de la tournée")
    df_display = st.session_state.df_opt[['street','zip','city','dist_step','cum_dist']].rename(
        columns={'street':'Adresse','zip':'CP','city':'Ville','dist_step':'Étape (km)','cum_dist':'Cumul (km)'}
    )
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.markdown('<div class="footer">🚛 Optimisateur de Tournées Logistiques • Gratuit</div>', unsafe_allow_html=True)

# --- Default Explanation ---
if not upl and not st.session_state.processed:
    st.markdown("""
    <div class="download">
    <b>Format attendu :</b><br>
    - Colonne Adresse ou Rue<br>
    - Colonne Code Postal<br>
    - Colonne Ville<br><br>
    <i>Le site détecte les colonnes, gère jusqu'à 250 adresses, détecte les hors secteur, et optimise votre tournée !</i>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">🚛 Optimisateur de Tournées Logistiques • 100% gratuit • Par Delestret Kim</div>', unsafe_allow_html=True)
