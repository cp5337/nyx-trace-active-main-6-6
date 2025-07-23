"""
KML Handler Module
----------------
Utilities for processing KML files to and from the application.

This module provides functions to:
1. Load and parse KML files into dataframes
2. Export location data to KML format for tools like Google Earth
3. Convert between various geospatial data formats
"""

import os
import pandas as pd
import xml.dom.minidom
from datetime import datetime
from fastkml import kml, styles
from shapely.geometry import Point
from geopy.geocoders import Nominatim

# Initialize geocoder for address to coordinates conversion
geocoder = Nominatim(user_agent="location_analysis_app")


def kml_to_dataframe(kml_file):
    """
    Parse a KML file into a pandas DataFrame

    Args:
        kml_file: Path to KML file or file-like object

    Returns:
        DataFrame with columns: Name, Description, Type, Latitude, Longitude
    """
    # Check if input is a file path or object
    if isinstance(kml_file, str):
        with open(kml_file, "rb") as f:
            doc = f.read()
    else:
        doc = kml_file.read()

    # Parse KML file
    k = kml.KML()
    k.from_string(doc)

    # Extract data from KML
    data = []

    # Helper function to extract features
    def extract_features(container):
        if hasattr(container, "features"):
            for feature in container.features():
                if hasattr(feature, "features"):
                    extract_features(feature)
                if (
                    hasattr(feature, "geometry")
                    and feature.geometry is not None
                ):
                    # Get point data
                    if feature.geometry.geom_type == "Point":
                        lon, lat = feature.geometry.coords[0]
                        data.append(
                            {
                                "Name": feature.name or "",
                                "Description": feature.description or "",
                                "Type": feature.type or "Placemark",
                                "Latitude": lat,
                                "Longitude": lon,
                                "Style": getattr(feature, "style_url", ""),
                            }
                        )

                    # For other geometry types like LineString, Polygon, etc.,
                    # we could add more processing here if needed

    # Process all features
    for feature in k.features():
        extract_features(feature)

    return pd.DataFrame(data)


def dataframe_to_kml(
    df,
    output_file=None,
    name_col="Name",
    desc_col="Description",
    lat_col="Latitude",
    lon_col="Longitude",
    type_col="Type",
):
    """
    Convert a DataFrame to KML format

    Args:
        df: DataFrame with location data
        output_file: Optional path to save KML file (if None, returns string)
        name_col: Column name for place names
        desc_col: Column name for descriptions
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        type_col: Column name for placemark type

    Returns:
        If output_file is None, returns KML as string
        Otherwise, saves to file and returns True
    """
    # Create KML document
    k = kml.KML()
    ns = "{http://www.opengis.net/kml/2.2}"

    # Create document
    d = kml.Document(
        ns, "doc", "Location Analysis Data", "Exported from Location Analysis"
    )
    k.append(d)

    # Create some styles
    styles_dict = {
        "Port": "#portStyle",
        "Border Crossing": "#borderStyle",
        "HIDTA Region": "#hidtaStyle",
        "Critical Infrastructure": "#infraStyle",
        "Incident": "#incidentStyle",
        "Default": "#defaultStyle",
    }

    # Add styles to document
    style_port = styles.Style(id="portStyle")
    style_port.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/shapes/sailing.png"
    )
    style_port.iconstyle.scale = 1.2
    d.append(style_port)

    style_border = styles.Style(id="borderStyle")
    style_border.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/shapes/road.png"
    )
    style_border.iconstyle.scale = 1.2
    d.append(style_border)

    style_hidta = styles.Style(id="hidtaStyle")
    style_hidta.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/shapes/police.png"
    )
    style_hidta.iconstyle.scale = 1.2
    d.append(style_hidta)

    style_infra = styles.Style(id="infraStyle")
    style_infra.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/shapes/square.png"
    )
    style_infra.iconstyle.scale = 1.2
    d.append(style_infra)

    style_incident = styles.Style(id="incidentStyle")
    style_incident.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/shapes/caution.png"
    )
    style_incident.iconstyle.scale = 1.2
    d.append(style_incident)

    style_default = styles.Style(id="defaultStyle")
    style_default.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png"
    )
    style_default.iconstyle.scale = 1.0
    d.append(style_default)

    # Create folder for placemarks
    fld = kml.Folder(
        ns, "fld", "Locations", "Location data exported from analysis dashboard"
    )
    d.append(fld)

    # Add placemarks from dataframe
    for _, row in df.iterrows():
        # Get values with fallbacks
        name = str(row.get(name_col, "Unnamed"))
        desc = str(row.get(desc_col, ""))
        lat = float(row.get(lat_col, 0))
        lon = float(row.get(lon_col, 0))
        ptype = str(row.get(type_col, "Default"))

        # Create placemark
        p = kml.Placemark(ns, name=name, description=desc)
        p.geometry = Point(lon, lat)

        # Assign style
        style_url = styles_dict.get(ptype, styles_dict["Default"])
        p.style_url = style_url

        fld.append(p)

    # Convert to KML string
    kml_str = k.to_string(prettyprint=True)

    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(kml_str)
        return True

    return kml_str


def geocode_address(address):
    """
    Convert address to latitude and longitude

    Args:
        address: Text address to geocode

    Returns:
        Tuple of (latitude, longitude) or None if geocoding failed
    """
    try:
        location = geocoder.geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Geocoding error: {str(e)}")

    return None


def batch_geocode_addresses(addresses):
    """
    Geocode multiple addresses

    Args:
        addresses: List of address strings

    Returns:
        DataFrame with Address, Latitude, Longitude columns
    """
    results = []

    for addr in addresses:
        coords = geocode_address(addr)
        if coords:
            lat, lon = coords
            results.append({"Address": addr, "Latitude": lat, "Longitude": lon})
        else:
            results.append(
                {"Address": addr, "Latitude": None, "Longitude": None}
            )

    return pd.DataFrame(results)


def addresses_to_kml(addresses, output_file=None, name_prefix="Location"):
    """
    Convert list of addresses directly to KML

    Args:
        addresses: List of address strings
        output_file: Optional path to save KML
        name_prefix: Prefix for naming placemarks

    Returns:
        KML string or True if saved to file
    """
    # Geocode addresses
    geo_df = batch_geocode_addresses(addresses)

    # Add Name column
    geo_df["Name"] = [f"{name_prefix} {i+1}" for i in range(len(geo_df))]
    geo_df["Description"] = geo_df["Address"]
    geo_df["Type"] = "Default"

    # Remove failed geocodes
    geo_df = geo_df.dropna(subset=["Latitude", "Longitude"])

    # Convert to KML
    return dataframe_to_kml(geo_df, output_file)
