import folium

def render_drone_map(drones):
    m = folium.Map(location=[39.0515, -94.5895], zoom_start=15)
    for d in drones:
        folium.Marker(
            [d["lat"], d["lon"]],
            tooltip=f"{d['drone_id']} ({d['model']})",
            icon=folium.Icon(color="blue", icon="plane", prefix="fa")
        ).add_to(m)
    m.save("drone_map.html")
    print("✅ Map saved to drone_map.html")