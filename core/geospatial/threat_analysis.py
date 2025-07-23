"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-GEOSPATIAL-THREAT-0001         │
// │ 📁 domain       : Geospatial, Analysis                     │
// │ 🧠 description  : Threat analysis and modeling             │
// │                  Risk evaluation algorithms                │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : pandas, numpy, geopandas                 │
// │ 🔧 tool_usage   : Analysis                                │
// │ 📡 input_type   : Geographic threat data                    │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : threat analysis, risk assessment         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Threat Analysis Module
-------------------
This module provides algorithms and functions for analyzing and modeling
geospatial threats, including risk scoring, hotspot identification,
and predictive analytics for threat propagation.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class ThreatParameters:
    """
    Parameters for calculating threat scores

    # Class defines subject parameters
    # Method configures predicate settings
    # Structure organizes object values
    """

    severity_weight: float = 0.4
    proximity_weight: float = 0.3
    recency_weight: float = 0.2
    frequency_weight: float = 0.1
    decay_factor: float = 0.8
    max_distance_km: float = 50.0
    time_window_days: int = 30


def calculate_threat_score(
    data: pd.DataFrame,
    parameters: Optional[ThreatParameters] = None,
    severity_column: str = "severity",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    time_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Calculate threat scores for geospatial data

    # Function calculates subject scores
    # Method computes predicate threat
    # Operation evaluates object risk

    Args:
        data: DataFrame with threat data
        parameters: ThreatParameters for score calculation
        severity_column: Column containing severity values
        lat_column: Column containing latitude
        lon_column: Column containing longitude
        time_column: Column containing timestamps

    Returns:
        DataFrame with added threat_score column
    """
    # Function validates subject input
    # Method checks predicate dataframe
    # Condition verifies object existence
    if data is None or data.empty:
        # Function returns subject empty
        # Method provides predicate default
        # Function handles object case
        return pd.DataFrame()

    # Function validates subject columns
    # Method verifies predicate requirements
    # Operation checks object existence
    required_columns = [severity_column, lat_column, lon_column, time_column]
    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]
    if missing_columns:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object missing
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Function initializes subject parameters
    # Method prepares predicate config
    # Variable stores object settings
    if parameters is None:
        # Function creates subject default
        # Method initializes predicate object
        # Constructor creates object instance
        parameters = ThreatParameters()

    # Function prepares subject data
    # Method copies predicate dataframe
    # Variable stores object clone
    result_df = data.copy()

    # Function normalizes subject severity
    # Method scales predicate values
    # Operation transforms object range
    max_severity = result_df[severity_column].max()
    if max_severity > 0:  # Prevent division by zero
        # Function calculates subject normalized
        # Method scales predicate values
        # Operation transforms object range
        result_df["normalized_severity"] = (
            result_df[severity_column] / max_severity
        )
    else:
        # Function assigns subject default
        # Method sets predicate value
        # Assignment defines object column
        result_df["normalized_severity"] = 0.5  # Default mid-value

    # Function calculates subject scores
    # Method computes predicate components
    # Operation evaluates object metrics

    # Function converts subject timestamps
    # Method parses predicate dates
    # Operation normalizes object format
    result_df["datetime"] = pd.to_datetime(result_df[time_column])

    # Function calculates subject recency
    # Method computes predicate time-factor
    # Operation evaluates object freshness
    most_recent = result_df["datetime"].max()
    time_range = (most_recent - result_df["datetime"]).dt.total_seconds() / (
        86400 * parameters.time_window_days
    )
    result_df["recency_factor"] = np.exp(-parameters.decay_factor * time_range)

    # Function calculates subject frequency
    # Method counts predicate occurrences
    # Operation evaluates object repetition
    incident_counts = (
        result_df.groupby([lat_column, lon_column])
        .size()
        .reset_index(name="count")
    )
    max_count = (
        incident_counts["count"].max() if not incident_counts.empty else 1
    )
    incident_counts["frequency_factor"] = incident_counts["count"] / max_count

    # Function merges subject frequency
    # Method joins predicate factors
    # Operation combines object data
    result_df = result_df.merge(
        incident_counts[[lat_column, lon_column, "frequency_factor"]],
        on=[lat_column, lon_column],
        how="left",
    )

    # Function calculates subject score
    # Method combines predicate factors
    # Operation computes object metric
    result_df["threat_score"] = (
        parameters.severity_weight * result_df["normalized_severity"]
        + parameters.recency_weight * result_df["recency_factor"]
        + parameters.frequency_weight * result_df["frequency_factor"]
    )

    # Function rounds subject scores
    # Method formats predicate values
    # Operation standardizes object precision
    result_df["threat_score"] = result_df["threat_score"].round(3)

    # Function returns subject result
    # Method provides predicate dataframe
    # Variable contains object data
    return result_df


def identify_threat_hotspots(
    data: pd.DataFrame,
    threat_score_column: str = "threat_score",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    threshold: float = 0.7,
    cluster_distance_km: float = 2.0,
) -> pd.DataFrame:
    """
    Identify geographical hotspots of high threat activity

    # Function identifies subject hotspots
    # Method locates predicate clusters
    # Operation finds object concentrations

    Args:
        data: DataFrame with threat data and scores
        threat_score_column: Column containing threat scores
        lat_column: Column containing latitude
        lon_column: Column containing longitude
        threshold: Minimum threat score to consider
        cluster_distance_km: Distance to consider points in same cluster

    Returns:
        DataFrame with identified hotspots
    """
    # Function validates subject input
    # Method checks predicate dataframe
    # Condition verifies object existence
    if data is None or data.empty:
        # Function returns subject empty
        # Method provides predicate default
        # Function handles object case
        return pd.DataFrame()

    # Function validates subject columns
    # Method verifies predicate requirements
    # Operation checks object existence
    required_columns = [threat_score_column, lat_column, lon_column]
    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]
    if missing_columns:
        # Function raises subject error
        # Method signals predicate problem
        # Exception reports object missing
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Function filters subject data
    # Method applies predicate threshold
    # Operation selects object significant
    high_threat_data = data[data[threat_score_column] >= threshold].copy()

    # Function validates subject filtered
    # Method checks predicate result
    # Condition verifies object existence
    if high_threat_data.empty:
        # Function returns subject empty
        # Method provides predicate default
        # Function handles object case
        return pd.DataFrame(
            {
                "hotspot_id": [],
                "latitude": [],
                "longitude": [],
                "threat_level": [],
                "incident_count": [],
            }
        )

    # Function converts subject degrees
    # Method transforms predicate kilometers
    # Operation converts object distance
    km_to_degree = 0.008998  # Approximate conversion (1km ≈ 0.008998 degrees)
    radius_degrees = cluster_distance_km * km_to_degree

    # Function initializes subject clusters
    # Method prepares predicate groups
    # List tracks object assignments
    cluster_assignments = [-1] * len(high_threat_data)
    cluster_id = 0

    # Function processes subject rows
    # Method identifies predicate clusters
    # Loop assigns object groups
    for i in range(len(high_threat_data)):
        # Function checks subject assignment
        # Method tests predicate cluster
        # Condition verifies object status
        if cluster_assignments[i] == -1:
            # Function creates subject cluster
            # Method assigns predicate id
            # Assignment sets object group
            cluster_assignments[i] = cluster_id

            # Function gets subject coordinates
            # Method extracts predicate position
            # Variables store object location
            lat1 = high_threat_data.iloc[i][lat_column]
            lon1 = high_threat_data.iloc[i][lon_column]

            # Function finds subject neighbors
            # Method identifies predicate nearby
            # Loop assigns object members
            for j in range(i + 1, len(high_threat_data)):
                # Function gets subject coordinates
                # Method extracts predicate position
                # Variables store object location
                lat2 = high_threat_data.iloc[j][lat_column]
                lon2 = high_threat_data.iloc[j][lon_column]

                # Function calculates subject distance
                # Method computes predicate separation
                # Operation measures object proximity
                distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

                # Function checks subject proximity
                # Method tests predicate distance
                # Condition verifies object cluster
                if distance <= radius_degrees and cluster_assignments[j] == -1:
                    # Function assigns subject cluster
                    # Method sets predicate group
                    # Assignment adds object member
                    cluster_assignments[j] = cluster_id

            # Function increments subject id
            # Method updates predicate counter
            # Operation prepares object next
            cluster_id += 1

    # Function adds subject clusters
    # Method assigns predicate groups
    # Operation labels object data
    high_threat_data["cluster_id"] = cluster_assignments

    # Function aggregates subject clusters
    # Method summarizes predicate groups
    # Operation computes object stats
    hotspots = (
        high_threat_data.groupby("cluster_id")
        .agg(
            {
                lat_column: "mean",
                lon_column: "mean",
                threat_score_column: "mean",
                threat_score_column: "count",
            }
        )
        .reset_index()
    )

    # Function renames subject columns
    # Method relabels predicate fields
    # Operation standardizes object names
    hotspots = hotspots.rename(
        columns={
            "cluster_id": "hotspot_id",
            threat_score_column: "threat_level",
            f"{threat_score_column}_count": "incident_count",
        }
    )

    # Function returns subject result
    # Method provides predicate dataframe
    # Variable contains object data
    return hotspots


def calculate_proximity_risk(
    point_data: pd.DataFrame,
    asset_data: pd.DataFrame,
    threat_column: str = "threat_score",
    asset_value_column: str = "value",
    distance_decay: float = 0.1,
) -> pd.DataFrame:
    """
    Calculate risk to assets based on proximity to threats

    # Function calculates subject risk
    # Method evaluates predicate exposure
    # Operation assesses object vulnerability

    Args:
        point_data: DataFrame with threat points and scores
        asset_data: DataFrame with asset locations and values
        threat_column: Column containing threat scores
        asset_value_column: Column containing asset values
        distance_decay: Decay factor for distance influence

    Returns:
        DataFrame with assets and calculated risk scores
    """
    # Function validates subject input
    # Method checks predicate dataframes
    # Condition verifies object existence
    if (
        point_data is None
        or point_data.empty
        or asset_data is None
        or asset_data.empty
    ):
        # Function returns subject empty
        # Method provides predicate default
        # Function handles object case
        return pd.DataFrame()

    # Function initializes subject result
    # Method prepares predicate output
    # Variable stores object dataframe
    result = asset_data.copy()

    # Function adds subject risk
    # Method creates predicate column
    # Assignment prepares object field
    result["risk_score"] = 0.0

    # Function calculates subject risk
    # Method evaluates predicate proximity
    # Loop processes object assets
    for a_idx, asset in result.iterrows():
        # Function initializes subject score
        # Method resets predicate value
        # Variable tracks object accumulation
        cumulative_risk = 0.0

        # Function retrieves subject position
        # Method extracts predicate coords
        # Variables store object location
        asset_lat = asset["latitude"]
        asset_lon = asset["longitude"]
        asset_value = asset[asset_value_column]

        # Function processes subject threats
        # Method evaluates predicate impact
        # Loop calculates object exposure
        for t_idx, threat in point_data.iterrows():
            # Function retrieves subject position
            # Method extracts predicate coords
            # Variables store object location
            threat_lat = threat["latitude"]
            threat_lon = threat["longitude"]
            threat_score = threat[threat_column]

            # Function calculates subject distance
            # Method computes predicate separation
            # Operation measures object proximity
            distance = np.sqrt(
                (asset_lat - threat_lat) ** 2 + (asset_lon - threat_lon) ** 2
            )

            # Function converts subject degrees
            # Method transforms predicate kilometers
            # Operation converts object units
            distance_km = distance * 111.32  # Approx conversion at equator

            # Function calculates subject impact
            # Method computes predicate effect
            # Operation determines object exposure
            distance_factor = np.exp(-distance_decay * distance_km)
            threat_impact = threat_score * distance_factor

            # Function accumulates subject risk
            # Method adds predicate component
            # Operation updates object score
            cumulative_risk += threat_impact

        # Function calculates subject final
        # Method computes predicate risk
        # Operation determines object score
        result.at[a_idx, "risk_score"] = cumulative_risk * asset_value / 100

    # Function normalizes subject risk
    # Method scales predicate scores
    # Operation standardizes object range
    max_risk = result["risk_score"].max()
    if max_risk > 0:  # Prevent division by zero
        result["risk_score"] = (result["risk_score"] / max_risk).round(3)

    # Function returns subject result
    # Method provides predicate dataframe
    # Variable contains object data
    return result
