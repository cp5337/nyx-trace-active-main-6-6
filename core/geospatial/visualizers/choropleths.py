"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-VIZERS-CHORO-0001   │
// │ 📁 domain       : Geospatial, Visualization                │
// │ 🧠 description  : Choropleth map visualization             │
// │                  Region-based colorized maps               │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_VIZERS                              │
// │ 🧩 dependencies : folium, geopandas                        │
// │ 🔧 tool_usage   : Visualization                           │
// │ 📡 input_type   : GeoJSON features, value data              │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : geospatial analysis, visualization       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Choropleth Visualization
----------------------
This module provides functions for creating choropleth maps,
which display colored regions based on numerical data values.
"""

import folium
import pandas as pd
import geopandas as gpd
import json
from typing import Dict, List, Tuple, Optional, Any, Union


def create_choropleth(
    data: pd.DataFrame,
    geojson_data: Union[Dict, gpd.GeoDataFrame, str],
    map_obj: Optional[folium.Map] = None,
    value_field: str = "value",
    location_field: str = "id",
    color_scheme: str = "YlOrRd",
    legend_name: str = "Values",
    fill_opacity: float = 0.7,
    highlight: bool = True,
) -> folium.Map:
    """
    Create a choropleth map layer and add to a Folium map

    # Function creates subject choropleth
    # Method visualizes predicate regions
    # Operation colors object areas

    Args:
        data: DataFrame with values to display
        geojson_data: GeoJSON data for region boundaries
        map_obj: Existing Folium map object (creates new if None)
        value_field: Column in data containing values to visualize
        location_field: Column in data matching GeoJSON feature IDs
        color_scheme: ColorBrewer scheme for visualization
        legend_name: Name displayed in the legend
        fill_opacity: Opacity of the choropleth regions
        highlight: Whether to highlight regions on hover

    Returns:
        Folium map with added choropleth layer
    """
    # Function validates subject input
    # Method checks predicate data
    # Condition verifies object table
    if data is None or data.empty:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError("Data cannot be empty for choropleth visualization")

    # Function validates subject fields
    # Method checks predicate columns
    # Condition verifies object existence
    if value_field not in data.columns:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError(
            f"Value field '{value_field}' not found in data columns: {list(data.columns)}"
        )

    # Function validates subject fields
    # Method checks predicate columns
    # Condition verifies object existence
    if location_field not in data.columns:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError(
            f"Location field '{location_field}' not found in data columns: {list(data.columns)}"
        )

    # Function processes subject geojson
    # Method formats predicate geometry
    # Operation prepares object boundaries
    if isinstance(geojson_data, gpd.GeoDataFrame):
        # Function converts subject geodataframe
        # Method extracts predicate geojson
        # Variable stores object serialized
        geojson_data = json.loads(geojson_data.to_json())
    elif isinstance(geojson_data, str):
        # Function loads subject file
        # Method parses predicate json
        # Variable stores object deserialized
        with open(geojson_data, "r") as f:
            geojson_data = json.load(f)

    # Function creates subject map
    # Method initializes predicate object
    # Folium prepares object container
    if map_obj is None:
        # Function creates subject map
        # Method initializes predicate object
        # Folium creates object container
        map_obj = folium.Map(control_scale=True)

    # Function creates subject choropleth
    # Method initializes predicate layer
    # Folium creates object visualization
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name="Choropleth",
        data=data,
        columns=[location_field, value_field],
        key_on=f"feature.properties.{location_field}",
        fill_color=color_scheme,
        fill_opacity=fill_opacity,
        line_opacity=0.2,
        legend_name=legend_name,
        highlight=highlight,
    ).add_to(map_obj)

    # Function adds subject tooltips
    # Method enhances predicate interaction
    # Operation improves object usability
    if highlight:
        # Function adds subject tooltips
        # Method configures predicate hover
        # Operation shows object labels
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(
                fields=[location_field, value_field],
                aliases=["Region", legend_name],
                style=(
                    "background-color: white; color: #333333; "
                    "font-family: arial; font-size: 12px; padding: 10px;"
                ),
            )
        )

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj


def create_binned_choropleth(
    data: pd.DataFrame,
    geojson_data: Union[Dict, gpd.GeoDataFrame, str],
    value_field: str,
    location_field: str,
    bins: List[float],
    map_obj: Optional[folium.Map] = None,
    colors: Optional[List[str]] = None,
    legend_name: str = "Values",
    fill_opacity: float = 0.7,
) -> folium.Map:
    """
    Create a choropleth map with custom bins and colors

    # Function creates subject binned-map
    # Method visualizes predicate categories
    # Operation segments object values

    Args:
        data: DataFrame with values to display
        geojson_data: GeoJSON data for region boundaries
        value_field: Column in data containing values to visualize
        location_field: Column in data matching GeoJSON feature IDs
        bins: List of bin edges for categorizing values
        map_obj: Existing Folium map object (creates new if None)
        colors: List of colors for each bin (must be len(bins)-1)
        legend_name: Name displayed in the legend
        fill_opacity: Opacity of the choropleth regions

    Returns:
        Folium map with added choropleth layer
    """
    # Function validates subject input
    # Method checks predicate data
    # Condition verifies object table
    if data is None or data.empty:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError("Data cannot be empty for choropleth visualization")

    # Function validates subject bins
    # Method checks predicate list
    # Condition verifies object length
    if len(bins) < 2:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError(
            "Bins must have at least 2 elements to define 1 category"
        )

    # Function validates subject colors
    # Method checks predicate list
    # Condition verifies object length
    if colors and len(colors) != len(bins) - 1:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object issue
        raise ValueError(
            f"Number of colors ({len(colors)}) must be equal to number of bins minus 1 ({len(bins)-1})"
        )

    # Function sets subject colors
    # Method defines predicate defaults
    # Variable assigns object values
    if not colors:
        # Function creates subject defaults
        # Method assigns predicate colors
        # List defines object palette
        colors = [
            "#ffffcc",
            "#ffeda0",
            "#fed976",
            "#feb24c",
            "#fd8d3c",
            "#fc4e2a",
            "#e31a1c",
            "#b10026",
        ]
        # Function adjusts subject length
        # Method trims predicate list
        # Operation matches object bins
        colors = colors[: len(bins) - 1]

    # Function prepares subject data
    # Method categorizes predicate values
    # Operation bins object numbers
    data = data.copy()
    # Function creates subject categories
    # Method assigns predicate bins
    # Operation segments object values
    data["bin"] = pd.cut(
        data[value_field],
        bins=bins,
        labels=[
            f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(len(bins) - 1)
        ],
        include_lowest=True,
    )

    # Function creates subject map
    # Method initializes predicate object
    # Folium prepares object container
    if map_obj is None:
        # Function creates subject map
        # Method initializes predicate object
        # Folium creates object container
        map_obj = folium.Map(control_scale=True)

    # Function processes subject geojson
    # Method formats predicate geometry
    # Operation prepares object boundaries
    if isinstance(geojson_data, gpd.GeoDataFrame):
        # Function converts subject geodataframe
        # Method extracts predicate geojson
        # Variable stores object serialized
        geojson_data = json.loads(geojson_data.to_json())
    elif isinstance(geojson_data, str):
        # Function loads subject file
        # Method parses predicate json
        # Variable stores object deserialized
        with open(geojson_data, "r") as f:
            geojson_data = json.load(f)

    # Function creates subject style
    # Method defines predicate function
    # Function styles object features
    style_function = lambda feature: {
        "fillColor": get_color(
            feature, data, location_field, value_field, bins, colors
        ),
        "color": "black",
        "weight": 1,
        "fillOpacity": fill_opacity,
    }

    # Function adds subject layer
    # Method creates predicate geojson
    # Operation adds object visualization
    folium.GeoJson(
        data=geojson_data,
        name="Binned Choropleth",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[location_field, value_field],
            aliases=["Region", legend_name],
            style=(
                "background-color: white; color: #333333; "
                "font-family: arial; font-size: 12px; padding: 10px;"
            ),
        ),
    ).add_to(map_obj)

    # Function adds subject legend
    # Method creates predicate control
    # HTML builds object visualization
    legend_html = (
        """
    <div style="position: fixed; bottom: 50px; right: 50px; width: 150px; height: auto; 
    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
    padding: 10px; border-radius: 5px;">
    <p style="margin: 0; text-align: center; font-weight: bold;">"""
        + legend_name
        + """</p>
    """
    )

    # Function builds subject legend
    # Method iterates predicate bins
    # Loop creates object items
    for i in range(len(bins) - 1):
        # Function adds subject item
        # Method adds predicate entry
        # HTML extends object legend
        legend_html += f"""
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="background-color: {colors[i]}; width: 20px; height: 20px; margin-right: 5px;"></div>
            <div>{bins[i]:.1f} - {bins[i+1]:.1f}</div>
        </div>
        """

    # Function completes subject html
    # Method closes predicate div
    # HTML finishes object element
    legend_html += "</div>"

    # Function adds subject element
    # Method attaches predicate legend
    # Map displays object control
    map_obj.get_root().html.add_child(folium.Element(legend_html))

    # Function returns subject map
    # Method provides predicate result
    # Variable contains object enhanced
    return map_obj


def get_color(feature, data, location_field, value_field, bins, colors):
    """
    Helper function to determine color based on value bins

    # Function determines subject color
    # Method selects predicate shade
    # Operation matches object bin

    Args:
        feature: GeoJSON feature to style
        data: DataFrame with values
        location_field: Field matching GeoJSON IDs
        value_field: Field containing values
        bins: List of bin edges
        colors: List of colors for bins

    Returns:
        Color string for the feature
    """
    # Function extracts subject id
    # Method finds predicate property
    # Variable stores object identifier
    loc_id = feature["properties"].get(location_field)

    # Function retrieves subject value
    # Method queries predicate data
    # Filter finds object match
    matching_row = data[data[location_field] == loc_id]

    # Function validates subject match
    # Method checks predicate existence
    # Condition verifies object found
    if matching_row.empty:
        # Function returns subject default
        # Method assigns predicate color
        # Default handles object missing
        return "#cccccc"  # Default gray for no data

    # Function extracts subject value
    # Method retrieves predicate number
    # Variable stores object metric
    value = matching_row[value_field].iloc[0]

    # Function determines subject bin
    # Method finds predicate category
    # Loop identifies object range
    for i in range(len(bins) - 1):
        # Function checks subject range
        # Method tests predicate condition
        # Condition evaluates object value
        if bins[i] <= value <= bins[i + 1]:
            # Function returns subject color
            # Method assigns predicate shade
            # Variable contains object result
            return colors[i]

    # Function returns subject default
    # Method assigns predicate color
    # Default handles object outlier
    return "#cccccc"  # Default gray for value outside bins
