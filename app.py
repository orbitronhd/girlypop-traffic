import streamlit as st
import cv2
import tempfile
import torch
import streamlit.components.v1 as components
from processor import TrafficProcessor
from gis_utils import create_dashboard_map, convert_to_geojson

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Autoflow GIS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');

        :root {
            --md-sys-color-primary: #D0BCFF;
            --md-sys-color-on-primary: #381E72;
            --md-sys-color-primary-container: #4F378B;
            --md-sys-color-surface: #141218;
            --md-sys-color-surface-variant: #49454F;
            --md-sys-color-outline: #938F99;
        }

        .stApp {
            background-color: var(--md-sys-color-surface);
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        }

        /* TYPOGRAPHY */
        h1, h2, h3 {
            font-family: 'Instrument Serif', serif !important;
            font-weight: 400 !important;
            letter-spacing: 0.05rem;
        }
        
        h1 {
            font-size: 3.5rem !important;
            background: linear-gradient(90deg, #EADDFF, #D0BCFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px !important;
        }

        p, div, label, button, .stMultiSelect, .stSelectbox, .stSlider {
            font-family: 'Atkinson Hyperlegible', sans-serif !important;
            color: #E6E1E5 !important;
        }

        /* COMPACT GLASS CARDS */
        div[data-testid="stMetric"], .glass-card, .stDataFrame {
            background: rgba(40, 35, 50, 0.4);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            padding: 10px 15px;
            transition: all 0.3s ease;
            min-height: auto;
        }
        
        div[data-testid="stMetric"]:hover {
            background: rgba(79, 55, 139, 0.2);
            border: 1px solid var(--md-sys-color-primary);
            transform: translateY(-2px);
        }

        div[data-testid="stMetric"] > div { width: 100% !important; }

        /* REVERTED EXPANDER STYLE (Glass Look) */
        div[data-testid="stExpander"] {
            background: rgba(20, 18, 24, 0.6);
            border-radius: 16px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* BUTTON STYLING */
        .stButton>button {
            background: rgba(208, 188, 255, 0.1) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #E6E1E5 !important;
            border-radius: 100px;
            font-weight: 600;
            padding: 0.25rem 1rem; 
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            font-size: 0.9rem;
        }

        .stButton>button:hover {
            background: rgba(208, 188, 255, 0.25) !important; 
            border: 1px solid rgba(208, 188, 255, 0.6) !important;
            box-shadow: 0 0 20px rgba(208, 188, 255, 0.3);
            transform: scale(1.02);
            color: #ffffff !important;
        }

        [data-testid="stMetricLabel"] { 
            opacity: 0.8; 
            font-size: 0.85rem !important; 
            margin-bottom: 0px !important;
        }
        [data-testid="stMetricValue"] { 
            color: var(--md-sys-color-primary) !important; 
            font-size: 1.8rem !important; 
        }

    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'counts' not in st.session_state:
    categories = ["Car", "Bike", "Bus", "Truck"]
    st.session_state.counts = {f"{d}_{c}": 0 for d in ["Incoming", "Outgoing"] for c in categories}
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = set()
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# --- HEADER ---
st.title("Autoflow GIS")

# --- COLLAPSIBLE SETTINGS BAR ---
with st.expander("⚙️ System Configuration", expanded=True):
    # Layout: Model | Threshold | GPU Status | Action Button
    c_conf1, c_conf2, c_conf3, c_conf4 = st.columns([1.5, 2, 1.5, 1])
    
    with c_conf1:
        model_type = st.selectbox("AI Model", ["yolov8n.pt", "yolov8m.pt"], index=0, label_visibility="collapsed")
    
    with c_conf2:
        conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.35, label_visibility="collapsed")

    with c_conf3:
        # GPU Indicator inside the config box
        if torch.cuda.is_available():
            st.markdown(f"<span style='color:#00ff00; font-weight:bold; font-size:0.9rem; line-height:2.5;'>⚡ GPU Mode (Fast)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#ffaa00; font-weight:bold; font-size:0.9rem; line-height:2.5;'>🐢 CPU Mode (Slow)</span>", unsafe_allow_html=True)
            
    with c_conf4:
        # Small Button
        if not st.session_state.processing_complete:
            stop_button = st.button("Stop", use_container_width=True)
        else:
            if st.button("Re-run", use_container_width=True):
                st.session_state.processing_complete = False
                st.rerun()

# --- MAIN TABS ---
tab_monitor, tab_gis = st.tabs(["Live Vision", "GIS Heatmap"])

with tab_monitor:
    col_video, col_stats = st.columns([2.5, 1]) 
    
    with col_video:
        video_file = st.file_uploader("Upload Footage", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")
        st_frame = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats:
        # Incoming Section
        st.markdown("### Incoming")
        c_in_1, c_in_2 = st.columns(2)
        m_in_car = c_in_1.empty()
        m_in_bike = c_in_2.empty()
        m_in_heavy = st.empty()
        
        st.write("") 
        
        # Outgoing Section
        st.markdown("### Outgoing")
        c_out_1, c_out_2 = st.columns(2)
        m_out_car = c_out_1.empty()
        m_out_bike = c_out_2.empty()
        m_out_heavy = st.empty()
        
        status_text = st.empty()
    
    if video_file:
        # 1. CHECK IF FILE IS NEW
        current_file_name = video_file.name
        if 'last_file' not in st.session_state or st.session_state.last_file != current_file_name:
            st.session_state.counts = {f"{d}_{c}": 0 for d in ["Incoming", "Outgoing"] for c in ["Car", "Bike", "Bus", "Truck"]}
            st.session_state.counted_ids = set()
            st.session_state.processing_complete = False 
            st.session_state.last_file = current_file_name

        # 2. RUN PROCESSING LOOP
        if not st.session_state.processing_complete:
            processor = TrafficProcessor(model_path=model_type, confidence=conf_threshold)
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            status_text.caption(f"Processing with {model_type}...")
            
            while cap.isOpened():
                if 'stop_button' in locals() and stop_button:
                    status_text.warning("Stopped.")
                    break
                    
                success, frame = cap.read()
                if not success:
                    status_text.success("Complete.")
                    break
                
                # PROCESS
                result = processor.process_frame(
                    frame, st.session_state.counts, st.session_state.counted_ids
                )
                
                if len(result) == 4:
                    annotated_frame, updated_counts, updated_ids, _ = result
                else:
                    annotated_frame, updated_counts, updated_ids = result
                
                st.session_state.counts = updated_counts
                st.session_state.counted_ids = updated_ids
                
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
                
                m_in_car.metric("Cars", st.session_state.counts["Incoming_Car"])
                m_in_bike.metric("Bikes", st.session_state.counts["Incoming_Bike"])
                m_in_heavy.metric("Heavy", st.session_state.counts["Incoming_Truck"] + st.session_state.counts["Incoming_Bus"])
                m_out_car.metric("Cars", st.session_state.counts["Outgoing_Car"])
                m_out_bike.metric("Bikes", st.session_state.counts["Outgoing_Bike"])
                m_out_heavy.metric("Heavy", st.session_state.counts["Outgoing_Truck"] + st.session_state.counts["Outgoing_Bus"])

            cap.release()
            st.session_state.processing_complete = True
            st.rerun()
            
        else:
            # 3. STATIC STATS
            status_text.success("Analysis Complete.")
            st_frame.info("Processing complete.")
            
            m_in_car.metric("Cars", st.session_state.counts["Incoming_Car"])
            m_in_bike.metric("Bikes", st.session_state.counts["Incoming_Bike"])
            m_in_heavy.metric("Heavy", st.session_state.counts["Incoming_Truck"] + st.session_state.counts["Incoming_Bus"])
            m_out_car.metric("Cars", st.session_state.counts["Outgoing_Car"])
            m_out_bike.metric("Bikes", st.session_state.counts["Outgoing_Bike"])
            m_out_heavy.metric("Heavy", st.session_state.counts["Outgoing_Truck"] + st.session_state.counts["Outgoing_Bus"])

with tab_gis:
    st.markdown("### 📍 Live Digital Heatmap")
    
    col_map, col_data = st.columns([3, 1])
    
    with col_map:
        total_cars = sum(st.session_state.counts.values())
        
        folium_map = create_dashboard_map(st.session_state.counts)
        map_html = folium_map._repr_html_()
        components.html(map_html, height=600)
        
    with col_data:
        st.markdown("#### Export Data")
        st.caption("Download vector data for ArcGIS/QGIS.")
        
        geojson_data = convert_to_geojson(st.session_state.counts)
        
        st.download_button(
            label="Download GeoJSON",
            data=geojson_data,
            file_name="traffic_data.json",
            mime="application/json",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)