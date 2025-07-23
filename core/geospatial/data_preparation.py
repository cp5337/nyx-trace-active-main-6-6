"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-DATAPREP-0001       │
// │ 📁 domain       : Geospatial, Data, Processing             │
// │ 🧠 description  : Geospatial data preparation utilities     │
// │                  Data validation and formatting            │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : pandas, geopandas                        │
// │ 🔧 tool_usage   : Data Processing                          │
// │ 📡 input_type   : DataFrames, coordinates                   │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data validation, transformation          │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Geospatial Data Preparation Utilities
-----------------------------------
This module provides functions for preparing, validating, and transforming
geospatial data for visualization and analysis.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Optional, Union, Any


def prepare_geospatial_data(data):
    """
    Process and validate geospatial data for visualization

    # Function processes subject geodata
    # Method prepares predicate visualization
    # Operation formats object coordinates
    # Code returns subject dataframe

    Args:
        data: Pandas DataFrame containing geospatial data

    Returns:
        Processed and validated DataFrame
    """
    # Function checks subject input
    # Method validates predicate data
    # Condition verifies object existence
    # Code ensures subject validity
    if data is None or data.empty:
        # Function creates subject empty
        # Method generates predicate placeholder
        # DataFrame creates object structure
        # Code returns subject default
        return pd.DataFrame({"latitude": [], "longitude": [], "intensity": []})

    # Function validates subject columns
    # Method checks predicate requirements
    # Condition verifies object coordinates
    # Code ensures subject structure
    required_cols = ["latitude", "longitude"]
    if not all(col in data.columns for col in required_cols):
        # Function maps subject aliases
        # Method defines predicate alternatives
        # Dictionary maps object synonyms
        # Code handles subject variations
        column_aliases = {
            "latitude": ["lat", "y", "latitude"],
            "longitude": ["lon", "long", "x", "longitude"],
        }

        # Function renames subject columns
        # Method maps predicate alternatives
        # Operation standardizes object names
        # Code normalizes subject format
        for std_col, aliases in column_aliases.items():
            # Function finds subject match
            # Method searches predicate columns
            # Filter identifies object alias
            # Code standardizes subject name
            for alias in aliases:
                if alias in data.columns and std_col not in data.columns:
                    # Function renames subject column
                    # Method standardizes predicate name
                    # DataFrame renames object field
                    # Code updates subject structure
                    data = data.rename(columns={alias: std_col})
                    break

    # Function validates subject success
    # Method checks predicate columns
    # Condition verifies object requirement
    # Code ensures subject structure
    if not all(col in data.columns for col in required_cols):
        # Function raises subject error
        # Method signals predicate problem
        # Exception indicates object format
        # Code halts subject processing
        raise ValueError(
            f"Data must contain coordinate columns. Found: {list(data.columns)}"
        )

    # Function handles subject intensity
    # Method checks predicate column
    # Condition verifies object optional
    # Code ensures subject completion
    if "intensity" not in data.columns:
        # Function adds subject column
        # Method extends predicate structure
        # DataFrame assigns object values
        # Code completes subject format
        data["intensity"] = 1.0  # Default intensity

    # Function validates subject coordinates
    # Method filters predicate values
    # Function removes object invalid
    # Code ensures subject quality
    data = data.dropna(subset=["latitude", "longitude"])

    # Function validates subject ranges
    # Method filters predicate values
    # Operation removes object invalid
    # Code ensures subject quality
    data = data[(data["latitude"] >= -90) & (data["latitude"] <= 90)]
    data = data[(data["longitude"] >= -180) & (data["longitude"] <= 180)]

    # Function returns subject result
    # Method provides predicate dataframe
    # Variable contains object data
    # Code delivers subject processed
    return data


def convert_to_geodataframe(df):
    """
    Convert a standard DataFrame with lat/lon to GeoDataFrame

    # Function converts subject dataframe
    # Method transforms predicate format
    # Operation creates object geodataframe

    Args:
        df: Pandas DataFrame with latitude and longitude columns

    Returns:
        GeoDataFrame with Point geometry
    """
    # Function validates subject input
    # Method checks predicate existence
    # Condition verifies object data
    if df is None or df.empty:
        return gpd.GeoDataFrame()

    # Function validates subject columns
    # Method checks predicate requirements
    # Condition verifies object coordinates
    required_cols = ["latitude", "longitude"]
    if not all(col in df.columns for col in required_cols):
        # Function processes subject data
        # Method prepares predicate format
        # Operation ensures object structure
        df = prepare_geospatial_data(df)

    # Function creates subject geometry
    # Method generates predicate points
    # Operation builds object shapes
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]

    # Function creates subject geodataframe
    # Method converts predicate format
    # Operation transforms object structure
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Function sets subject CRS
    # Method defines predicate projection
    # Operation configures object coordinates
    gdf.crs = "EPSG:4326"

    # Function returns subject result
    # Method provides predicate geodataframe
    # Variable contains object data
    return gdf


def extract_coordinates(gdf):
    """
    Extract coordinates from a GeoDataFrame's Point geometry

    # Function extracts subject coordinates
    # Method retrieves predicate positions
    # Operation obtains object locations

    Args:
        gdf: GeoDataFrame with Point geometry

    Returns:
        DataFrame with extracted latitude and longitude
    """
    # Function validates subject input
    # Method checks predicate existence
    # Condition verifies object data
    if gdf is None or gdf.empty or not isinstance(gdf, gpd.GeoDataFrame):
        return pd.DataFrame(columns=["latitude", "longitude"])

    # Function creates subject lists
    # Method prepares predicate containers
    # Operation initializes object storage
    latitudes = []
    longitudes = []

    # Function iterates subject geometry
    # Method processes predicate points
    # Operation extracts object coordinates
    for point in gdf.geometry:
        # Function validates subject type
        # Method checks predicate point
        # Condition verifies object geometry
        if point is None:
            latitudes.append(None)
            longitudes.append(None)
            continue

        # Function extracts subject coordinates
        # Method retrieves predicate values
        # Operation obtains object position
        longitudes.append(point.x)
        latitudes.append(point.y)

    # Function creates subject result
    # Method updates predicate dataframe
    # Operation adds object columns
    result = gdf.copy()
    result["latitude"] = latitudes
    result["longitude"] = longitudes

    # Function returns subject result
    # Method provides predicate dataframe
    # Variable contains object data
    return result
