import folium
from folium.plugins import HeatMap
import pandas as pd
import random

def create_dashboard_map(counts):
    """
    Generates a heatmap based on TOTAL counts.
    ROBUST METHOD: If counts > 0, it creates fake GPS points around Kochi.
    This guarantees a visual result.
    """
    # 1. Base Location (Kochi)
    center_lat, center_long = 9.9312, 76.2673
    
    # 2. Map Setup
    m = folium.Map(location=[center_lat, center_long], zoom_start=18, tiles="CartoDB dark_matter")
    
    # 3. GENERATE HEATMAP DATA
    # Calculate total traffic
    total_vehicles = sum(counts.values())
    
    heatmap_data = []
    
    # Always add 5 "ghost" points so the map isn't empty on startup
    simulated_traffic = total_vehicles + 5
    
    for _ in range(simulated_traffic):
        # Gaussian distribution creates a nice "blob" around the center
        fake_lat = random.gauss(center_lat, 0.0006)
        fake_lon = random.gauss(center_long, 0.0006)
        
        # [Lat, Lon, Intensity]
        heatmap_data.append([fake_lat, fake_lon, 1.0])

    # 4. Add Heatmap Layer
    if heatmap_data:
        HeatMap(
            heatmap_data, 
            radius=25,
            blur=15, 
            min_opacity=0.4,
            gradient={0.4: 'cyan', 0.65: 'lime', 1: 'red'}
        ).add_to(m)

    # 5. Context Marker
    folium.Marker(
        [center_lat, center_long],
        popup=f"<b>Sensor Node A</b><br>Traffic Count: {total_vehicles}",
        icon=folium.Icon(color="red", icon="video", prefix="fa")
    ).add_to(m)
    
    return m

def convert_to_geojson(counts):
    df = pd.DataFrame([counts])
    return df.to_json()