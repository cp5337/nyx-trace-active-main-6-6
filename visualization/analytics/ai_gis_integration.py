"""
AI-GIS Integration Module
-----------------------
This module provides integration between AI models and GIS systems,
allowing for intelligent analysis of geospatial data, pattern recognition,
and predictive modeling for the NyxTrace platform.
"""

import os
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import time
import base64
from io import BytesIO
import openai
import google.generativeai as genai
import wolframalpha
from pathlib import Path
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_gis_integration')

# Default prompt templates
PROMPT_TEMPLATES = {
    "terrain_analysis": """
Analyze the terrain features in this geographic area:
Coordinates: {coordinates}
Area Description: {description}

Identify key terrain features that would be advantageous for:
1. Surveillance positions
2. Movement corridors
3. Defensive positions
4. Areas providing concealment
5. Potential hazards

Use the provided elevation data, slope information, and land cover to inform your analysis.
Provide reasoning for each identified feature.
""",
    "pattern_detection": """
Analyze the following geospatial data points:
{data_points}

Area of interest: {area}
Time period: {time_period}

Identify any significant patterns, clusters, or anomalies in this data.
Specifically look for:
1. Temporal patterns
2. Spatial clusters
3. Correlations with known infrastructure or features
4. Unusual outliers or anomalies
5. Potential causal relationships

Provide confidence levels for each pattern identified.
""",
    "route_analysis": """
Analyze the potential routes between these points:
Starting point: {start_point}
Destination: {end_point}
Terrain information: {terrain_info}

Consider factors including:
1. Terrain constraints (slopes, water features, barriers)
2. Concealment options
3. Speed of movement
4. Detection risk
5. Alternative approaches

Identify the most likely routes and explain the tradeoffs between them.
""",
    "activity_prediction": """
Based on the historical patterns in this area:
{historical_data}

And the current context:
{current_context}

Predict likely future activities in these regions over the next {time_frame}.
Consider:
1. Seasonal or cyclical patterns
2. Recent changes in activity
3. Known causal factors
4. Anomalies that may indicate new patterns
5. Confidence levels for predictions

Provide specific geographic predictions if possible.
"""
}


class AIGIS:
    """
    AI-GIS Integration class for geospatial intelligence analysis
    
    This class provides methods for:
    - Processing geospatial data with AI models
    - Analyzing terrain and infrastructure
    - Detecting patterns and anomalies
    - Predicting activities and movements
    - Representing GIS data in graph structures
    """
    
    def __init__(self):
        """Initialize AI-GIS integration"""
        # Load API keys from environment if available
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        self.xai_api_key = os.environ.get('XAI_API_KEY')
        self.google_ai_api_key = os.environ.get('GOOGLE_AI_API_KEY')
        self.wolfram_app_id = os.environ.get('WOLFRAM_APP_ID')
        self.mapbox_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
        
        # Initialize clients
        self._initialize_clients()
        
        # Keep track of available models
        self.available_models = self._detect_available_models()
        
        logger.info("AI-GIS Integration initialized")
        if len(self.available_models) > 0:
            logger.info(f"Available AI models: {', '.join(self.available_models)}")
        else:
            logger.warning("No AI models available - using default models where possible")
    
    def _initialize_clients(self):
        """Initialize API clients based on available credentials"""
        # Initialize OpenAI if API key available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found, client not initialized")
        
        # Initialize Google AI if API key available
        if self.google_ai_api_key:
            genai.configure(api_key=self.google_ai_api_key)
            self.genai_client = genai
            logger.info("Google Generative AI client initialized")
        else:
            self.genai_client = None
            logger.warning("Google AI API key not found, client not initialized")
        
        # Initialize Wolfram Alpha if app ID available
        if self.wolfram_app_id:
            self.wolfram_client = wolframalpha.Client(self.wolfram_app_id)
            logger.info("Wolfram Alpha client initialized")
        else:
            self.wolfram_client = None
            logger.warning("Wolfram Alpha App ID not found, client not initialized")
        
        # For Anthropic and xAI, we'd need their official Python clients
        # Here we just log that they're not available
        if not self.anthropic_api_key:
            logger.warning("Anthropic API key not found")
        
        if not self.xai_api_key:
            logger.warning("xAI API key not found")
    
    def _detect_available_models(self) -> List[str]:
        """Detect which AI models are available based on API keys"""
        available = []
        
        if self.openai_client:
            available.append("openai-gpt4")
        
        if self.genai_client:
            available.append("google-gemini")
        
        if self.anthropic_api_key:
            available.append("anthropic-claude")
        
        if self.xai_api_key:
            available.append("xai-grok")
        
        if self.wolfram_client:
            available.append("wolfram-alpha")
        
        return available
    
    def analyze_terrain(self, 
                       dem_data: Union[str, gpd.GeoDataFrame], 
                       area_description: str,
                       coordinates: Optional[str] = None,
                       model: str = "auto") -> Dict[str, Any]:
        """
        Analyze terrain features using AI
        
        Args:
            dem_data: Path to DEM file or GeoDataFrame with terrain data
            area_description: Description of the area
            coordinates: Optional coordinates string (e.g. "34.05, -118.25")
            model: AI model to use ("auto" selects best available)
            
        Returns:
            Dictionary with terrain analysis results
        """
        # If coordinates not provided, try to extract from dem_data if it's a GeoDataFrame
        if coordinates is None and isinstance(dem_data, gpd.GeoDataFrame):
            if not dem_data.empty:
                bounds = dem_data.total_bounds
                coordinates = f"{bounds[1]:.4f}, {bounds[0]:.4f} to {bounds[3]:.4f}, {bounds[2]:.4f}"
        
        # Prepare the prompt
        prompt = PROMPT_TEMPLATES["terrain_analysis"].format(
            coordinates=coordinates or "Not provided",
            description=area_description
        )
        
        # Select model if auto
        if model == "auto":
            model = self._select_best_model(task="terrain_analysis")
        
        # Process with selected model
        result = self._process_with_model(prompt, model)
        
        # Additional terrain-specific post-processing could be done here
        
        return {
            "analysis": result,
            "model_used": model,
            "prompt": prompt,
            "coordinates": coordinates,
            "timestamp": time.time()
        }
    
    def detect_patterns(self, 
                       data: Union[pd.DataFrame, gpd.GeoDataFrame],
                       area_name: str,
                       time_period: str,
                       model: str = "auto") -> Dict[str, Any]:
        """
        Detect patterns in geospatial data
        
        Args:
            data: DataFrame or GeoDataFrame with geospatial data
            area_name: Name of the area being analyzed
            time_period: Time period of the data (e.g. "Jan 2023 - Mar 2023")
            model: AI model to use ("auto" selects best available)
            
        Returns:
            Dictionary with pattern detection results
        """
        # Convert data to text representation for the prompt
        if isinstance(data, gpd.GeoDataFrame) or isinstance(data, pd.DataFrame):
            # Get a sample of data points (limit to 20 for prompt size)
            sample_size = min(20, len(data))
            if len(data) > sample_size:
                sampled_data = data.sample(sample_size)
            else:
                sampled_data = data
            
            # Create a text representation of the data
            data_text = "Sample data points:\n"
            for idx, row in sampled_data.iterrows():
                data_text += f"- Point {idx}: "
                
                # Include geometry if available
                if hasattr(row, 'geometry') and row.geometry is not None:
                    if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                        data_text += f"Coordinates ({row.geometry.y:.4f}, {row.geometry.x:.4f}), "
                
                # Include other key columns
                for col in row.index:
                    if col != 'geometry' and not col.startswith('_'):
                        data_text += f"{col}: {row[col]}, "
                
                data_text = data_text.rstrip(", ") + "\n"
            
            # Add summary statistics
            data_text += f"\nTotal data points: {len(data)}\n"
            
            # Add time information if available
            time_cols = [col for col in data.columns if any(t in col.lower() for t in ['time', 'date', 'timestamp'])]
            if time_cols:
                time_col = time_cols[0]
                earliest = data[time_col].min()
                latest = data[time_col].max()
                data_text += f"Time range: {earliest} to {latest}\n"
        else:
            # If not a dataframe, convert to string
            data_text = str(data)
        
        # Prepare the prompt
        prompt = PROMPT_TEMPLATES["pattern_detection"].format(
            data_points=data_text,
            area=area_name,
            time_period=time_period
        )
        
        # Select model if auto
        if model == "auto":
            model = self._select_best_model(task="pattern_detection")
        
        # Process with selected model
        result = self._process_with_model(prompt, model)
        
        return {
            "analysis": result,
            "model_used": model,
            "prompt": prompt,
            "data_points": len(data) if hasattr(data, '__len__') else 'unknown',
            "timestamp": time.time()
        }
    
    def analyze_routes(self, 
                      start_point: Tuple[float, float], 
                      end_point: Tuple[float, float],
                      terrain_info: Union[str, gpd.GeoDataFrame],
                      model: str = "auto") -> Dict[str, Any]:
        """
        Analyze potential routes between points
        
        Args:
            start_point: Starting coordinates (lat, lon)
            end_point: Ending coordinates (lat, lon)
            terrain_info: Terrain information as text or GeoDataFrame
            model: AI model to use ("auto" selects best available)
            
        Returns:
            Dictionary with route analysis results
        """
        # Convert terrain_info to text if it's a GeoDataFrame
        if isinstance(terrain_info, gpd.GeoDataFrame):
            terrain_text = f"Terrain includes {len(terrain_info)} features.\n"
            
            # Get bounds
            bounds = terrain_info.total_bounds
            terrain_text += f"Area bounds: ({bounds[1]:.4f}, {bounds[0]:.4f}) to ({bounds[3]:.4f}, {bounds[2]:.4f})\n"
            
            # Describe geometry types
            geom_types = terrain_info.geometry.type.value_counts().to_dict()
            terrain_text += "Geometry types: "
            for gt, count in geom_types.items():
                terrain_text += f"{gt}: {count}, "
            terrain_text = terrain_text.rstrip(", ") + "\n"
            
            # Include some feature properties if available
            if len(terrain_info.columns) > 1:  # More than just geometry
                terrain_text += "Feature properties include:\n"
                for col in terrain_info.columns:
                    if col != 'geometry':
                        terrain_text += f"- {col}\n"
        else:
            terrain_text = str(terrain_info)
        
        # Prepare the prompt
        prompt = PROMPT_TEMPLATES["route_analysis"].format(
            start_point=f"{start_point[0]:.4f}, {start_point[1]:.4f}",
            end_point=f"{end_point[0]:.4f}, {end_point[1]:.4f}",
            terrain_info=terrain_text
        )
        
        # Select model if auto
        if model == "auto":
            model = self._select_best_model(task="route_analysis")
        
        # Process with selected model
        result = self._process_with_model(prompt, model)
        
        return {
            "analysis": result,
            "model_used": model,
            "prompt": prompt,
            "start_point": start_point,
            "end_point": end_point,
            "timestamp": time.time()
        }
    
    def predict_activity(self, 
                        historical_data: Union[pd.DataFrame, gpd.GeoDataFrame, str],
                        current_context: str,
                        time_frame: str,
                        model: str = "auto") -> Dict[str, Any]:
        """
        Predict future activities based on historical data
        
        Args:
            historical_data: Historical data as DataFrame or text
            current_context: Current contextual situation
            time_frame: Time frame for predictions (e.g. "next 7 days")
            model: AI model to use ("auto" selects best available)
            
        Returns:
            Dictionary with activity predictions
        """
        # Convert historical_data to text if it's a DataFrame
        if isinstance(historical_data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Summarize the data
            data_text = f"Historical data with {len(historical_data)} records.\n"
            
            # Get time range if available
            time_cols = [col for col in historical_data.columns if any(t in col.lower() for t in ['time', 'date', 'timestamp'])]
            if time_cols:
                time_col = time_cols[0]
                earliest = historical_data[time_col].min()
                latest = historical_data[time_col].max()
                data_text += f"Time range: {earliest} to {latest}\n"
            
            # Summarize key columns
            data_text += "Data includes columns: "
            data_text += ", ".join(historical_data.columns.tolist()) + "\n"
            
            # Add some sample data
            sample_size = min(10, len(historical_data))
            if len(historical_data) > 0:
                data_text += "\nSample records:\n"
                for idx, row in historical_data.sample(sample_size).iterrows():
                    data_text += f"Record {idx}: "
                    for col in row.index:
                        if col != 'geometry' and not col.startswith('_'):
                            data_text += f"{col}: {row[col]}, "
                    data_text = data_text.rstrip(", ") + "\n"
        else:
            data_text = str(historical_data)
        
        # Prepare the prompt
        prompt = PROMPT_TEMPLATES["activity_prediction"].format(
            historical_data=data_text,
            current_context=current_context,
            time_frame=time_frame
        )
        
        # Select model if auto
        if model == "auto":
            model = self._select_best_model(task="activity_prediction")
        
        # Process with selected model
        result = self._process_with_model(prompt, model)
        
        return {
            "prediction": result,
            "model_used": model,
            "prompt": prompt,
            "time_frame": time_frame,
            "timestamp": time.time()
        }
    
    def analyze_image(self, 
                     image_path: str, 
                     query: str = "Analyze this geospatial image and identify key features",
                     model: str = "auto") -> Dict[str, Any]:
        """
        Analyze a geospatial image using AI vision capabilities
        
        Args:
            image_path: Path to image file
            query: Query or prompt for the analysis
            model: AI model to use ("auto" selects best available)
            
        Returns:
            Dictionary with image analysis results
        """
        # Check if the image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {
                "error": "Image file not found",
                "path": image_path,
                "timestamp": time.time()
            }
        
        # Select appropriate model for image analysis
        if model == "auto":
            model = self._select_best_model(task="image_analysis")
        
        # Encode image as base64 for API requests
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error reading image file: {str(e)}")
            return {
                "error": f"Error reading image file: {str(e)}",
                "path": image_path,
                "timestamp": time.time()
            }
        
        # Process with selected model
        if "openai" in model and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Using o because this requires vision capabilities
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1024
                )
                result = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error with OpenAI image analysis: {str(e)}")
                result = f"Error: {str(e)}"
        
        elif "google-gemini" in model and self.genai_client:
            try:
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content(
                    [query, {"inlineData": {"data": base64_image, "mimeType": "image/jpeg"}}]
                )
                result = response.text
            except Exception as e:
                logger.error(f"Error with Google Gemini image analysis: {str(e)}")
                result = f"Error: {str(e)}"
        
        else:
            result = "No suitable multimodal model available for image analysis. Please provide an OpenAI or Google API key."
        
        return {
            "analysis": result,
            "model_used": model,
            "query": query,
            "image_path": image_path,
            "timestamp": time.time()
        }
    
    def calculate_mathematical_terrain(self, 
                                     query: str, 
                                     data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use Wolfram Alpha to perform mathematical terrain calculations
        
        Args:
            query: Mathematical query for Wolfram Alpha
            data: Optional additional data for the calculation
            
        Returns:
            Dictionary with calculation results
        """
        if not self.wolfram_client:
            logger.warning("Wolfram Alpha client not initialized")
            return {
                "error": "Wolfram Alpha client not initialized. Please provide a Wolfram App ID.",
                "query": query,
                "timestamp": time.time()
            }
        
        # Add data to query if provided
        full_query = query
        if data:
            # Format additional data for the query
            for key, value in data.items():
                full_query += f", {key}={value}"
        
        try:
            # Query Wolfram Alpha
            res = self.wolfram_client.query(full_query)
            
            # Process results
            results = {}
            for pod in res.pods:
                title = pod.title
                results[title] = []
                for sub in pod.subpods:
                    if hasattr(sub, 'plaintext') and sub.plaintext:
                        results[title].append(sub.plaintext)
            
            return {
                "results": results,
                "query": full_query,
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error(f"Error with Wolfram Alpha calculation: {str(e)}")
            return {
                "error": f"Error with Wolfram Alpha calculation: {str(e)}",
                "query": full_query,
                "timestamp": time.time()
            }
    
    def gis_to_graph(self, 
                   data: gpd.GeoDataFrame, 
                   relationship_type: str = "proximity",
                   threshold: float = 0.01) -> nx.Graph:
        """
        Convert GIS data to a graph representation for network analysis
        
        Args:
            data: GeoDataFrame with geospatial objects
            relationship_type: Type of relationship between objects
                Options: 'proximity', 'connectivity', 'hierarchy'
            threshold: Distance threshold for establishing edges
            
        Returns:
            NetworkX graph representation of GIS data
        """
        if data.empty:
            logger.warning("Empty GeoDataFrame provided for graph conversion")
            return nx.Graph()
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes for each geometry
        for idx, row in data.iterrows():
            # Create attributes from row data
            attrs = {col: row[col] for col in row.index if col != 'geometry'}
            
            # Add centroid coordinates
            if hasattr(row.geometry, 'centroid'):
                attrs['x'] = row.geometry.centroid.x
                attrs['y'] = row.geometry.centroid.y
            
            # Add node
            G.add_node(idx, **attrs)
        
        # Add edges based on relationship type
        if relationship_type == 'proximity':
            # Create edges between geometries that are within threshold distance
            for i, row_i in data.iterrows():
                for j, row_j in data.iterrows():
                    if i != j:
                        try:
                            dist = row_i.geometry.distance(row_j.geometry)
                            if dist <= threshold:
                                G.add_edge(i, j, weight=dist, relationship='proximity')
                        except Exception as e:
                            logger.error(f"Error calculating distance between {i} and {j}: {str(e)}")
        
        elif relationship_type == 'connectivity':
            # Create edges based on shared boundaries
            for i, row_i in data.iterrows():
                for j, row_j in data.iterrows():
                    if i != j:
                        try:
                            if row_i.geometry.touches(row_j.geometry):
                                # Calculate the length of the shared boundary
                                boundary_length = row_i.geometry.intersection(row_j.geometry.boundary).length
                                G.add_edge(i, j, weight=boundary_length, relationship='connectivity')
                        except Exception as e:
                            logger.error(f"Error checking connectivity between {i} and {j}: {str(e)}")
        
        elif relationship_type == 'hierarchy':
            # Create edges based on containment
            for i, row_i in data.iterrows():
                for j, row_j in data.iterrows():
                    if i != j:
                        try:
                            if row_i.geometry.contains(row_j.geometry):
                                area_ratio = row_j.geometry.area / row_i.geometry.area
                                G.add_edge(i, j, weight=area_ratio, relationship='contains')
                            elif row_j.geometry.contains(row_i.geometry):
                                area_ratio = row_i.geometry.area / row_j.geometry.area
                                G.add_edge(j, i, weight=area_ratio, relationship='contains')
                        except Exception as e:
                            logger.error(f"Error checking containment between {i} and {j}: {str(e)}")
        
        else:
            logger.warning(f"Unknown relationship type: {relationship_type}")
        
        return G
    
    def analyze_graph(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Analyze a graph representation of geospatial data
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary with graph analysis results
        """
        results = {}
        
        # Basic graph properties
        results["node_count"] = G.number_of_nodes()
        results["edge_count"] = G.number_of_edges()
        results["density"] = nx.density(G)
        
        # Connected components
        connected_components = list(nx.connected_components(G))
        results["connected_components"] = len(connected_components)
        
        # Calculate centrality measures
        if G.number_of_nodes() > 0:
            try:
                results["degree_centrality"] = nx.degree_centrality(G)
                results["betweenness_centrality"] = nx.betweenness_centrality(G)
                results["closeness_centrality"] = nx.closeness_centrality(G)
            except Exception as e:
                logger.error(f"Error calculating centrality measures: {str(e)}")
        
        # Find important nodes
        if G.number_of_nodes() > 0:
            # Find nodes with highest degree centrality
            degree_centrality = nx.degree_centrality(G)
            results["highest_degree_nodes"] = sorted(degree_centrality.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:5]
            
            # Find nodes with highest betweenness centrality
            if "betweenness_centrality" in results:
                betweenness_centrality = results["betweenness_centrality"]
                results["highest_betweenness_nodes"] = sorted(betweenness_centrality.items(), 
                                                            key=lambda x: x[1], 
                                                            reverse=True)[:5]
        
        # Compute clustering coefficient
        try:
            results["clustering_coefficient"] = nx.average_clustering(G)
        except Exception as e:
            logger.error(f"Error calculating clustering coefficient: {str(e)}")
        
        # Detect communities
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G)
            results["community_count"] = len(communities)
            results["community_sizes"] = [len(c) for c in communities]
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
        
        return results
    
    def export_graph_to_cytoscape(self, G: nx.Graph, output_file: str) -> bool:
        """
        Export a graph to Cytoscape-compatible format
        
        Args:
            G: NetworkX graph
            output_file: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, nothing to export")
            return False
        
        try:
            # Prepare data structure for Cytoscape.js
            cy_elements = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for node_id, node_data in G.nodes(data=True):
                node_element = {
                    "data": {
                        "id": str(node_id)
                    }
                }
                
                # Add all attributes
                for attr, value in node_data.items():
                    # Ensure attribute is JSON serializable
                    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                        node_element["data"][attr] = value
                
                # Set position if available
                if 'x' in node_data and 'y' in node_data:
                    node_element["position"] = {
                        "x": node_data['x'],
                        "y": node_data['y']
                    }
                
                cy_elements["nodes"].append(node_element)
            
            # Add edges
            for source, target, edge_data in G.edges(data=True):
                edge_element = {
                    "data": {
                        "id": f"e_{source}_{target}",
                        "source": str(source),
                        "target": str(target)
                    }
                }
                
                # Add all attributes
                for attr, value in edge_data.items():
                    # Ensure attribute is JSON serializable
                    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                        edge_element["data"][attr] = value
                
                cy_elements["edges"].append(edge_element)
            
            # Write to file
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(cy_elements, f, indent=2)
            
            logger.info(f"Graph exported to Cytoscape format: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting graph to Cytoscape format: {str(e)}")
            return False
    
    def export_to_google_earth(self, 
                             data: gpd.GeoDataFrame, 
                             output_file: str,
                             name_column: Optional[str] = None,
                             description_columns: Optional[List[str]] = None,
                             style: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export geospatial data to KML format for Google Earth
        
        Args:
            data: GeoDataFrame with geospatial objects
            output_file: Path to output KML file
            name_column: Column to use for placemark names
            description_columns: Columns to include in placemark descriptions
            style: Optional styling for KML elements
            
        Returns:
            True if successful, False otherwise
        """
        if data.empty:
            logger.warning("Empty GeoDataFrame, nothing to export")
            return False
        
        try:
            # Convert to WGS84 if not already
            if data.crs != "EPSG:4326":
                data = data.to_crs("EPSG:4326")
            
            # Export to KML
            if output_file.endswith('.kml'):
                # Set driver options
                driver_options = {}
                
                # Set name field
                if name_column and name_column in data.columns:
                    driver_options['NameField'] = name_column
                
                # Set description fields
                if description_columns:
                    desc_cols = [col for col in description_columns if col in data.columns]
                    if desc_cols:
                        data['description'] = data.apply(
                            lambda row: '<br>'.join([f"{col}: {row[col]}" for col in desc_cols]),
                            axis=1
                        )
                        driver_options['DescriptionField'] = 'description'
                
                # Export to KML
                data.to_file(output_file, driver='KML', **driver_options)
                logger.info(f"Data exported to KML: {output_file}")
                return True
                
            # If output is KMZ, we need to create KML first, then zip it
            elif output_file.endswith('.kmz'):
                # Create temporary KML file
                temp_kml = output_file.replace('.kmz', '_temp.kml')
                
                # Export to KML first
                result = self.export_to_google_earth(
                    data, 
                    temp_kml, 
                    name_column, 
                    description_columns, 
                    style
                )
                
                if result:
                    # Create KMZ (zip file with KML)
                    import zipfile
                    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as kmz:
                        kmz.write(temp_kml, os.path.basename(temp_kml))
                    
                    # Remove temporary KML
                    os.remove(temp_kml)
                    
                    logger.info(f"Data exported to KMZ: {output_file}")
                    return True
                else:
                    return False
            else:
                logger.error(f"Output file must be .kml or .kmz: {output_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to Google Earth format: {str(e)}")
            return False
    
    def _select_best_model(self, task: str) -> str:
        """
        Select the best available model for a given task
        
        Args:
            task: Task type
            
        Returns:
            Model identifier
        """
        # Default preferences by task
        task_preferences = {
            "terrain_analysis": ["openai-gpt4", "anthropic-claude", "google-gemini", "xai-grok"],
            "pattern_detection": ["openai-gpt4", "anthropic-claude", "google-gemini", "xai-grok"],
            "route_analysis": ["openai-gpt4", "anthropic-claude", "google-gemini", "xai-grok"],
            "activity_prediction": ["openai-gpt4", "anthropic-claude", "google-gemini", "xai-grok"],
            "image_analysis": ["openai-gpt4", "google-gemini"],
            "mathematical_analysis": ["wolfram-alpha", "openai-gpt4", "google-gemini"]
        }
        
        # Get preferences for the task
        preferences = task_preferences.get(task, ["openai-gpt4", "google-gemini", "anthropic-claude", "xai-grok"])
        
        # Find the first available model in order of preference
        for model in preferences:
            if model in self.available_models:
                return model
        
        # If no preferred model is available, default to local/built-in options
        return "built-in"
    
    def _process_with_model(self, prompt: str, model: str) -> str:
        """
        Process a prompt with the specified AI model
        
        Args:
            prompt: Prompt text
            model: Model to use
            
        Returns:
            Response text
        """
        # Process with OpenAI
        if model == "openai-gpt4" and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Default to newest model
                    messages=[
                        {"role": "system", "content": "You are a geospatial intelligence analysis assistant. Provide detailed, accurate analysis of geographic data with a focus on practical insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error with OpenAI processing: {str(e)}")
                return f"Error processing with OpenAI: {str(e)}"
        
        # Process with Google Gemini
        elif model == "google-gemini" and self.genai_client:
            try:
                gemini_model = self.genai_client.GenerativeModel('gemini-pro')
                response = gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Error with Google Gemini processing: {str(e)}")
                return f"Error processing with Google Gemini: {str(e)}"
        
        # Process with built-in methods (placeholder)
        elif model == "built-in":
            logger.warning("Using built-in processing (limited capabilities)")
            return f"Analysis of your request:\n\nThe NyxTrace system has processed your query using built-in methods, but for full AI analysis capabilities, please provide API keys for OpenAI, Google Gemini, or other supported models.\n\nYour prompt was: {prompt[:100]}..."
        
        # No suitable model available
        else:
            return "No suitable AI model available for processing. Please provide API keys for OpenAI, Google, or other supported models."