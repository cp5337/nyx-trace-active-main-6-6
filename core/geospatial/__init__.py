"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-INIT-0001           │
// │ 📁 domain       : Geospatial, Core                         │
// │ 🧠 description  : Geospatial core package                  │
// │                  Mapping and visualization tools           │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : folium, geopandas, pandas                │
// │ 🔧 tool_usage   : Geospatial Analysis                      │
// │ 📡 input_type   : Geographic data                          │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : mapping, visualization                    │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Geospatial Core Package
---------------------
This package provides geospatial analysis and visualization tools,
including base maps, data preparation utilities, plugin management,
threat analysis, reporting, and specialized visualizers for heatmaps,
markers, choropleths, and networks.
"""

from .base_maps import (
    create_base_map,
    create_advanced_map,
    add_tile_layers,
    add_legend,
    MAP_TILES,
)

from .data_preparation import (
    prepare_geospatial_data,
    convert_to_geodataframe,
    extract_coordinates,
)

from .plugin_manager import (
    initialize_plugins,
    get_plugin_status,
    PLUGIN_SUPPORT,
)

from .threat_analysis import (
    ThreatParameters,
    calculate_threat_score,
    identify_threat_hotspots,
    calculate_proximity_risk,
)

from .reporting import (
    generate_threat_summary,
    create_threat_time_series,
    create_threat_histogram,
    generate_geospatial_report,
    export_report_as_json,
    get_report_download_link,
    display_report_summary,
)

# Import visualization components
from .visualizers import (
    create_heatmap,
    render_heatmap,
    add_markers,
    create_marker_clusters,
    create_choropleth,
    create_network_graph,
)
