"""
PostGIS/Supabase Integration Module
--------------------------------
This module provides integration with PostGIS and Supabase for spatial
database functionality in the NyxTrace platform.
"""

import os
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from geoalchemy2 import Geometry, WKTElement, functions as gfunc
from shapely.geometry import shape, Point, LineString, Polygon, MultiPolygon
from shapely.wkt import loads as wkt_loads
from datetime import datetime
import supabase
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('postgis_integration')

# SQLAlchemy base
Base = declarative_base()


# Define SQLAlchemy models for GIS data
class GeoFeature(Base):
    """Base model for geographic features"""
    __tablename__ = 'geo_features'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    feature_type = Column(String(50))
    properties = Column(Text)  # JSON string of properties
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Geometry column - can be any geometry type
    geom = Column(Geometry('GEOMETRY', srid=4326))


class SpatialDatabaseIntegration:
    """
    Spatial database integration using PostGIS and Supabase
    
    This class provides methods for:
    - Connecting to PostGIS databases (direct or via Supabase)
    - Storing and retrieving geospatial data
    - Performing spatial queries and analysis
    - Converting between GeoJSON, GeoDataFrames, and database records
    """
    
    def __init__(self, 
                connection_type: str = "direct",
                connection_string: Optional[str] = None,
                supabase_url: Optional[str] = None,
                supabase_key: Optional[str] = None):
        """
        Initialize spatial database integration
        
        Args:
            connection_type: Type of connection ("direct" or "supabase")
            connection_string: Connection string for direct PostgreSQL connection
            supabase_url: Supabase URL
            supabase_key: Supabase API key
        """
        self.connection_type = connection_type
        self.connection_string = connection_string
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        
        # Engine and session
        self.engine = None
        self.Session = None
        self.supabase_client = None
        
        # Try to get from environment if not provided
        if not self.connection_string and connection_type == "direct":
            self.connection_string = os.environ.get('DATABASE_URL')
        
        if not self.supabase_url and connection_type == "supabase":
            self.supabase_url = os.environ.get('SUPABASE_URL')
        
        if not self.supabase_key and connection_type == "supabase":
            self.supabase_key = os.environ.get('SUPABASE_KEY')
        
        # Connect to database
        self._connect()
    
    def _connect(self) -> bool:
        """
        Connect to spatial database
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if self.connection_type == "direct":
                if not self.connection_string:
                    logger.error("No connection string provided for direct connection")
                    return False
                
                # Create SQLAlchemy engine
                self.engine = create_engine(self.connection_string)
                
                # Create session factory
                self.Session = sessionmaker(bind=self.engine)
                
                # Test connection
                with self.engine.connect() as conn:
                    # Check if PostGIS extension is installed
                    result = conn.execute(text("SELECT PostGIS_version();"))
                    postgis_version = result.scalar()
                    logger.info(f"Connected to PostGIS database (version: {postgis_version})")
                
                return True
                
            elif self.connection_type == "supabase":
                if not self.supabase_url or not self.supabase_key:
                    logger.error("Supabase URL and key required for Supabase connection")
                    return False
                
                # Create Supabase client
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                
                # Test connection by fetching version
                # This requires RLS policies allowing this query or admin rights
                response = self.supabase_client.rpc('get_postgis_version').execute()
                
                if hasattr(response, 'data'):
                    logger.info(f"Connected to Supabase with PostGIS (version: {response.data})")
                else:
                    logger.info("Connected to Supabase, but could not verify PostGIS version")
                
                return True
                
            else:
                logger.error(f"Unknown connection type: {self.connection_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to spatial database: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """
        Create database tables
        
        Returns:
            True if tables created successfully, False otherwise
        """
        if self.connection_type == "direct" and self.engine:
            try:
                # Create all tables
                Base.metadata.create_all(self.engine)
                logger.info("Created database tables")
                return True
            except Exception as e:
                logger.error(f"Error creating tables: {str(e)}")
                return False
        else:
            logger.error("Cannot create tables - no direct database connection")
            return False
    
    def store_geodataframe(self, 
                         gdf: gpd.GeoDataFrame, 
                         table_name: str, 
                         if_exists: str = 'replace',
                         geom_col: str = 'geometry') -> bool:
        """
        Store a GeoDataFrame in the spatial database
        
        Args:
            gdf: GeoDataFrame to store
            table_name: Name of the table to store in
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            geom_col: Name of the geometry column
            
        Returns:
            True if stored successfully, False otherwise
        """
        if gdf.empty:
            logger.warning("Empty GeoDataFrame, nothing to store")
            return False
        
        # Make sure GeoDataFrame is in WGS84
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        try:
            if self.connection_type == "direct" and self.engine:
                # Convert geometry to WKT
                gdf_copy = gdf.copy()
                gdf_copy['geom'] = gdf_copy[geom_col].apply(lambda x: WKTElement(x.wkt, srid=4326))
                
                # Drop original geometry column
                gdf_copy = gdf_copy.drop(columns=[geom_col])
                
                # Write to PostGIS
                gdf_copy.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
                
                # Add geometry index
                with self.engine.connect() as conn:
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_geom ON {table_name} USING GIST (geom);"))
                    conn.commit()
                
                logger.info(f"Stored GeoDataFrame to table {table_name}")
                return True
                
            elif self.connection_type == "supabase" and self.supabase_client:
                # For Supabase, we need to convert to GeoJSON and use the RPC approach
                
                # Convert to GeoJSON
                geo_json = json.loads(gdf.to_json())
                
                # Process in batches of 100 to avoid payload size issues
                features = geo_json['features']
                batch_size = 100
                
                for i in range(0, len(features), batch_size):
                    batch = features[i:i+batch_size]
                    
                    # Create batch GeoJSON
                    batch_geojson = {
                        "type": "FeatureCollection",
                        "features": batch
                    }
                    
                    # Send to Supabase using RPC
                    response = self.supabase_client.rpc(
                        'import_geojson',
                        {
                            'geojson': json.dumps(batch_geojson),
                            'table_name': table_name,
                            'srid': 4326,
                            'if_exists': if_exists
                        }
                    ).execute()
                    
                    # If first batch and if_exists is 'replace', set to 'append' for subsequent batches
                    if if_exists == 'replace':
                        if_exists = 'append'
                
                logger.info(f"Stored GeoDataFrame to Supabase table {table_name}")
                return True
                
            else:
                logger.error("No database connection available")
                return False
                
        except Exception as e:
            logger.error(f"Error storing GeoDataFrame: {str(e)}")
            return False
    
    def query_to_geodataframe(self, 
                            query: str, 
                            params: Optional[Dict[str, Any]] = None,
                            geom_col: str = 'geom') -> gpd.GeoDataFrame:
        """
        Execute a spatial query and return results as GeoDataFrame
        
        Args:
            query: SQL query with geometry column
            params: Query parameters
            geom_col: Name of the geometry column
            
        Returns:
            GeoDataFrame with query results
        """
        try:
            if self.connection_type == "direct" and self.engine:
                # Execute query
                with self.engine.connect() as conn:
                    # Convert geometry to WKB in query
                    if 'ST_AsEWKB' not in query and 'ST_AsBinary' not in query:
                        # Add conversion for geometry column
                        query = query.replace(f"SELECT {geom_col}", f"SELECT ST_AsEWKB({geom_col}) as {geom_col}")
                        query = query.replace(f"SELECT * ", f"SELECT *, ST_AsEWKB({geom_col}) as {geom_col} ")
                    
                    result = conn.execute(text(query), params or {})
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    if df.empty:
                        return gpd.GeoDataFrame(columns=[geom_col], geometry=geom_col, crs="EPSG:4326")
                    
                    # Convert WKB to shapely geometry
                    import shapely.wkb as wkb
                    df[geom_col] = df[geom_col].apply(lambda x: wkb.loads(x) if x else None)
                    
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326")
                    
                    return gdf
                    
            elif self.connection_type == "supabase" and self.supabase_client:
                # For Supabase, we need to use RPC
                response = self.supabase_client.rpc(
                    'execute_spatial_query',
                    {
                        'query_text': query,
                        'params': json.dumps(params or {})
                    }
                ).execute()
                
                if not hasattr(response, 'data') or not response.data:
                    return gpd.GeoDataFrame(columns=[geom_col], geometry=geom_col, crs="EPSG:4326")
                
                # Convert response to DataFrame
                df = pd.DataFrame(response.data)
                
                if geom_col not in df.columns:
                    logger.warning(f"Geometry column '{geom_col}' not found in query results")
                    return gpd.GeoDataFrame(df, crs="EPSG:4326")
                
                # Convert GeoJSON to shapely geometry
                df[geom_col] = df[geom_col].apply(
                    lambda x: shape(json.loads(x)) if x else None
                )
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326")
                
                return gdf
                
            else:
                logger.error("No database connection available")
                return gpd.GeoDataFrame()
                
        except Exception as e:
            logger.error(f"Error executing spatial query: {str(e)}")
            return gpd.GeoDataFrame()
    
    def spatial_join(self,
                    left_table: str,
                    right_table: str,
                    join_type: str = 'ST_Intersects',
                    left_geom: str = 'geom',
                    right_geom: str = 'geom',
                    select_columns: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """
        Perform a spatial join between two tables
        
        Args:
            left_table: Name of the left table
            right_table: Name of the right table
            join_type: Spatial relationship ('ST_Intersects', 'ST_Contains', etc.)
            left_geom: Name of the geometry column in left table
            right_geom: Name of the geometry column in right table
            select_columns: Columns to select (defaults to all)
            
        Returns:
            GeoDataFrame with joined data
        """
        # Build column list
        if select_columns:
            # Make sure geometry columns are included
            if left_geom not in select_columns:
                select_columns.append(f"a.{left_geom}")
            
            columns = ", ".join([f"a.{col}" if '.' not in col else col for col in select_columns])
        else:
            columns = f"a.*, b.*"
        
        # Build query
        query = f"""
        SELECT {columns}
        FROM {left_table} a
        JOIN {right_table} b
        ON {join_type}(a.{left_geom}, b.{right_geom})
        """
        
        # Execute query
        return self.query_to_geodataframe(query, geom_col=left_geom)
    
    def buffer_analysis(self, 
                       table: str, 
                       distance: float, 
                       geom_col: str = 'geom',
                       output_table: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Perform a buffer analysis on a table
        
        Args:
            table: Name of the table
            distance: Buffer distance in meters
            geom_col: Name of the geometry column
            output_table: Optional name of output table to store results
            
        Returns:
            GeoDataFrame with buffered geometries
        """
        # Build query
        query = f"""
        SELECT id, name, ST_Buffer({geom_col}::geography, {distance})::geometry as {geom_col}
        FROM {table}
        """
        
        # Execute query
        result = self.query_to_geodataframe(query, geom_col=geom_col)
        
        # Store results if output table specified
        if output_table and not result.empty:
            self.store_geodataframe(result, output_table)
        
        return result
    
    def distance_matrix(self, 
                       table: str, 
                       geom_col: str = 'geom',
                       where_clause: str = '') -> pd.DataFrame:
        """
        Calculate distance matrix between features in a table
        
        Args:
            table: Name of the table
            geom_col: Name of the geometry column
            where_clause: Optional WHERE clause for filtering
            
        Returns:
            DataFrame with distance matrix
        """
        # Build query
        where_stmt = f"WHERE {where_clause}" if where_clause else ""
        query = f"""
        WITH points AS (
            SELECT id, name, {geom_col}
            FROM {table}
            {where_stmt}
        )
        SELECT 
            a.id as id_a, 
            a.name as name_a, 
            b.id as id_b, 
            b.name as name_b,
            ST_Distance(a.{geom_col}::geography, b.{geom_col}::geography) as distance_meters
        FROM points a
        CROSS JOIN points b
        WHERE a.id != b.id
        ORDER BY a.id, b.id
        """
        
        # Execute query
        if self.connection_type == "direct" and self.engine:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
                
        elif self.connection_type == "supabase" and self.supabase_client:
            response = self.supabase_client.rpc(
                'execute_query',
                {'query_text': query}
            ).execute()
            
            if hasattr(response, 'data') and response.data:
                df = pd.DataFrame(response.data)
                return df
            else:
                return pd.DataFrame()
        else:
            logger.error("No database connection available")
            return pd.DataFrame()
    
    def nearest_neighbor_analysis(self, 
                                point_table: str, 
                                neighbor_table: str,
                                k: int = 5,
                                point_geom: str = 'geom',
                                neighbor_geom: str = 'geom',
                                max_distance: Optional[float] = None) -> pd.DataFrame:
        """
        Find k nearest neighbors for each point
        
        Args:
            point_table: Table with points to find neighbors for
            neighbor_table: Table with potential neighbors
            k: Number of nearest neighbors
            point_geom: Geometry column in point table
            neighbor_geom: Geometry column in neighbor table
            max_distance: Maximum distance in meters (optional)
            
        Returns:
            DataFrame with nearest neighbors
        """
        # Build distance constraint
        distance_clause = f"AND ST_Distance(p.{point_geom}::geography, n.{neighbor_geom}::geography) <= {max_distance}" if max_distance else ""
        
        # Build query
        query = f"""
        SELECT 
            p.id as point_id, 
            p.name as point_name, 
            n.id as neighbor_id, 
            n.name as neighbor_name,
            ST_Distance(p.{point_geom}::geography, n.{neighbor_geom}::geography) as distance_meters
        FROM {point_table} p
        CROSS JOIN LATERAL (
            SELECT id, name, {neighbor_geom}
            FROM {neighbor_table} n
            WHERE n.id != p.id
            {distance_clause}
            ORDER BY p.{point_geom} <-> n.{neighbor_geom}
            LIMIT {k}
        ) n
        ORDER BY p.id, distance_meters
        """
        
        # Execute query
        if self.connection_type == "direct" and self.engine:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
                
        elif self.connection_type == "supabase" and self.supabase_client:
            response = self.supabase_client.rpc(
                'execute_query',
                {'query_text': query}
            ).execute()
            
            if hasattr(response, 'data') and response.data:
                df = pd.DataFrame(response.data)
                return df
            else:
                return pd.DataFrame()
        else:
            logger.error("No database connection available")
            return pd.DataFrame()
    
    def load_geojson(self, 
                   geojson_file: str, 
                   table_name: str, 
                   if_exists: str = 'replace',
                   id_column: Optional[str] = None) -> bool:
        """
        Load GeoJSON file into the database
        
        Args:
            geojson_file: Path to GeoJSON file
            table_name: Table to load into
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            id_column: Optional ID column name in GeoJSON
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Read GeoJSON file
            with open(geojson_file, 'r') as f:
                geojson = json.load(f)
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geojson['features'])
            
            # Set ID column if specified
            if id_column and id_column in gdf.columns:
                gdf = gdf.set_index(id_column)
            
            # Store in database
            return self.store_geodataframe(gdf, table_name, if_exists=if_exists)
            
        except Exception as e:
            logger.error(f"Error loading GeoJSON: {str(e)}")
            return False
    
    def execute_query(self, 
                     query: str, 
                     params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute a raw SQL query
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            True if executed successfully, False otherwise
        """
        try:
            if self.connection_type == "direct" and self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text(query), params or {})
                    conn.commit()
                logger.info("Executed SQL query successfully")
                return True
                
            elif self.connection_type == "supabase" and self.supabase_client:
                response = self.supabase_client.rpc(
                    'execute_query',
                    {
                        'query_text': query,
                        'params': json.dumps(params or {})
                    }
                ).execute()
                
                logger.info("Executed SQL query via Supabase RPC")
                return True
                
            else:
                logger.error("No database connection available")
                return False
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return False
    
    def export_to_geojson(self, 
                        table_name: str, 
                        output_file: str,
                        where_clause: str = '',
                        geom_col: str = 'geom') -> bool:
        """
        Export table to GeoJSON file
        
        Args:
            table_name: Table to export
            output_file: Output GeoJSON file
            where_clause: Optional WHERE clause
            geom_col: Geometry column name
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            # Build query
            where_stmt = f"WHERE {where_clause}" if where_clause else ""
            query = f"SELECT * FROM {table_name} {where_stmt}"
            
            # Execute query
            gdf = self.query_to_geodataframe(query, geom_col=geom_col)
            
            # Export to GeoJSON
            if not gdf.empty:
                gdf.to_file(output_file, driver='GeoJSON')
                logger.info(f"Exported table {table_name} to GeoJSON: {output_file}")
                return True
            else:
                logger.warning(f"No data to export from table {table_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to GeoJSON: {str(e)}")
            return False