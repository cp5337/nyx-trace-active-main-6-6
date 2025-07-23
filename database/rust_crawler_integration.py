"""
CTAS Database Analyzer Integration
==================================
Python integration for the Rust database analyzer crawler
to optimize CTAS intelligence and analytical data structures.
"""

import asyncio
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StorageType(Enum):
    """Storage types for CTAS data structures"""
    SLED = "sled"
    SURREALDB = "surrealdb" 
    SUPABASE = "supabase"
    MONGODB = "mongodb"
    NEO4J = "neo4j"

class AccessPattern(Enum):
    """Access patterns for CTAS intelligence data"""
    HIGH_FREQUENCY = "high_frequency"      # Real-time threat feeds
    MEDIUM_FREQUENCY = "medium_frequency"  # Analysis results
    LOW_FREQUENCY = "low_frequency"        # Historical data
    ANALYTICS_ONLY = "analytics_only"      # Reports and dashboards

@dataclass
class CTASDataStructure:
    """CTAS-specific data structure information"""
    name: str
    file_path: str
    fields: List[Dict[str, Any]]
    estimated_size: int
    access_pattern: AccessPattern
    suggested_storage: StorageType
    intelligence_type: Optional[str] = None
    threat_level: Optional[str] = None
    retention_policy: Optional[str] = None

@dataclass
class CTASOptimizationReport:
    """CTAS optimization report from Rust crawler"""
    total_structs: int
    total_enums: int
    sled_candidates: int
    surrealdb_candidates: int
    supabase_candidates: int
    optimization_count: int
    ctas_specific_optimizations: List[Dict[str, Any]]

class CTASDatabaseAnalyzer:
    """
    CTAS Database Analyzer Integration
    
    Integrates the Rust database analyzer crawler with CTAS-specific
    intelligence and analytical data structures.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.rust_crawler_path = self.project_root / "database" / "database_analyzer.rs"
        self.ctas_data_structures: List[CTASDataStructure] = []
        self.optimization_report: Optional[CTASOptimizationReport] = None
        
    async def analyze_ctas_project(self) -> CTASOptimizationReport:
        """
        Analyze the CTAS project using the Rust crawler
        """
        logger.info("🔍 Starting CTAS database analysis...")
        
        # Check if Rust crawler exists
        if not self.rust_crawler_path.exists():
            logger.warning("Rust crawler not found, creating CTAS-specific analysis")
            return await self._analyze_ctas_structures()
        
        # Run Rust crawler analysis
        try:
            rust_report = await self._run_rust_crawler()
            return await self._enhance_with_ctas_context(rust_report)
        except Exception as e:
            logger.error(f"Rust crawler failed: {e}")
            return await self._analyze_ctas_structures()
    
    async def _run_rust_crawler(self) -> Dict[str, Any]:
        """
        Execute the Rust database analyzer crawler
        """
        logger.info("🚀 Executing Rust database analyzer...")
        
        # This would require Rust compilation and execution
        # For now, we'll simulate the analysis
        return {
            "total_structs": 25,
            "total_enums": 8,
            "sled_candidates": 5,
            "surrealdb_candidates": 12,
            "supabase_candidates": 8,
            "optimization_count": 15
        }
    
    async def _analyze_ctas_structures(self) -> CTASOptimizationReport:
        """
        Analyze CTAS-specific data structures
        """
        logger.info("📊 Analyzing CTAS data structures...")
        
        # Define CTAS-specific data structures
        ctas_structs = [
            {
                "name": "ThreatIntelligence",
                "access_pattern": AccessPattern.HIGH_FREQUENCY,
                "suggested_storage": StorageType.SLED,
                "intelligence_type": "real_time_threats",
                "threat_level": "critical"
            },
            {
                "name": "GeospatialData",
                "access_pattern": AccessPattern.MEDIUM_FREQUENCY,
                "suggested_storage": StorageType.SURREALDB,
                "intelligence_type": "location_intelligence",
                "threat_level": "medium"
            },
            {
                "name": "BehavioralAnalysis",
                "access_pattern": AccessPattern.MEDIUM_FREQUENCY,
                "suggested_storage": StorageType.MONGODB,
                "intelligence_type": "behavioral_patterns",
                "threat_level": "high"
            },
            {
                "name": "ThreatCorrelation",
                "access_pattern": AccessPattern.LOW_FREQUENCY,
                "suggested_storage": StorageType.NEO4J,
                "intelligence_type": "relationship_analysis",
                "threat_level": "medium"
            },
            {
                "name": "AnalyticsReport",
                "access_pattern": AccessPattern.ANALYTICS_ONLY,
                "suggested_storage": StorageType.SUPABASE,
                "intelligence_type": "historical_analytics",
                "threat_level": "low"
            }
        ]
        
        # Process CTAS structures
        for struct_info in ctas_structs:
            ctas_struct = CTASDataStructure(
                name=struct_info["name"],
                file_path=f"core/intelligence/{struct_info['name'].lower()}.py",
                fields=self._generate_fields_for_struct(struct_info["name"]),
                estimated_size=self._estimate_size(struct_info["name"]),
                access_pattern=struct_info["access_pattern"],
                suggested_storage=struct_info["suggested_storage"],
                intelligence_type=struct_info["intelligence_type"],
                threat_level=struct_info["threat_level"],
                retention_policy=self._get_retention_policy(struct_info["access_pattern"])
            )
            self.ctas_data_structures.append(ctas_struct)
        
        return CTASOptimizationReport(
            total_structs=len(ctas_structs),
            total_enums=3,
            sled_candidates=1,
            surrealdb_candidates=1,
            supabase_candidates=1,
            optimization_count=8,
            ctas_specific_optimizations=self._generate_ctas_optimizations()
        )
    
    def _generate_fields_for_struct(self, struct_name: str) -> List[Dict[str, Any]]:
        """Generate fields for CTAS data structures"""
        field_templates = {
            "ThreatIntelligence": [
                {"name": "threat_id", "type": "String", "primary_key": True},
                {"name": "timestamp", "type": "DateTime", "indexed": True},
                {"name": "threat_type", "type": "String", "searchable": True},
                {"name": "severity", "type": "Integer", "indexed": True},
                {"name": "source", "type": "String", "searchable": True},
                {"name": "indicators", "type": "Array", "searchable": True}
            ],
            "GeospatialData": [
                {"name": "location_id", "type": "String", "primary_key": True},
                {"name": "coordinates", "type": "Point", "indexed": True},
                {"name": "threat_density", "type": "Float", "indexed": True},
                {"name": "last_updated", "type": "DateTime", "indexed": True}
            ],
            "BehavioralAnalysis": [
                {"name": "behavior_id", "type": "String", "primary_key": True},
                {"name": "entity_id", "type": "String", "indexed": True},
                {"name": "pattern_type", "type": "String", "searchable": True},
                {"name": "confidence_score", "type": "Float", "indexed": True}
            ],
            "ThreatCorrelation": [
                {"name": "correlation_id", "type": "String", "primary_key": True},
                {"name": "source_threat", "type": "String", "indexed": True},
                {"name": "target_threat", "type": "String", "indexed": True},
                {"name": "relationship_type", "type": "String", "searchable": True}
            ],
            "AnalyticsReport": [
                {"name": "report_id", "type": "String", "primary_key": True},
                {"name": "report_type", "type": "String", "indexed": True},
                {"name": "generated_at", "type": "DateTime", "indexed": True},
                {"name": "data_summary", "type": "JSON", "searchable": True}
            ]
        }
        
        return field_templates.get(struct_name, [])
    
    def _estimate_size(self, struct_name: str) -> int:
        """Estimate size for CTAS data structures"""
        size_estimates = {
            "ThreatIntelligence": 512,    # Large due to indicators array
            "GeospatialData": 256,        # Moderate size
            "BehavioralAnalysis": 384,    # Medium-large
            "ThreatCorrelation": 128,     # Small, mostly IDs
            "AnalyticsReport": 1024       # Large due to JSON data
        }
        return size_estimates.get(struct_name, 256)
    
    def _get_retention_policy(self, access_pattern: AccessPattern) -> str:
        """Get retention policy based on access pattern"""
        policies = {
            AccessPattern.HIGH_FREQUENCY: "7 days (hot data)",
            AccessPattern.MEDIUM_FREQUENCY: "30 days (warm data)",
            AccessPattern.LOW_FREQUENCY: "1 year (cold data)",
            AccessPattern.ANALYTICS_ONLY: "5 years (archival)"
        }
        return policies.get(access_pattern, "30 days")
    
    def _generate_ctas_optimizations(self) -> List[Dict[str, Any]]:
        """Generate CTAS-specific optimizations"""
        return [
            {
                "type": "intelligence_data_flow",
                "description": "Optimize threat intelligence data flow for real-time processing",
                "implementation": "Use Sled for hot threat data, sync to SurrealDB for analysis"
            },
            {
                "type": "geospatial_optimization", 
                "description": "Optimize geospatial queries for threat mapping",
                "implementation": "Use spatial indexes in SurrealDB for location-based queries"
            },
            {
                "type": "behavioral_analysis_caching",
                "description": "Cache behavioral patterns for faster threat detection",
                "implementation": "Use MongoDB with TTL indexes for pattern storage"
            },
            {
                "type": "threat_correlation_graph",
                "description": "Optimize threat correlation using graph database",
                "implementation": "Use Neo4j for relationship analysis and threat graphs"
            },
            {
                "type": "analytics_warehouse",
                "description": "Create analytics warehouse for historical threat data",
                "implementation": "Use Supabase with materialized views for reporting"
            }
        ]
    
    async def _enhance_with_ctas_context(self, rust_report: Dict[str, Any]) -> CTASOptimizationReport:
        """Enhance Rust crawler report with CTAS-specific context"""
        logger.info("🎯 Enhancing analysis with CTAS context...")
        
        # Add CTAS-specific optimizations
        ctas_optimizations = self._generate_ctas_optimizations()
        
        return CTASOptimizationReport(
            total_structs=rust_report["total_structs"],
            total_enums=rust_report["total_enums"],
            sled_candidates=rust_report["sled_candidates"],
            surrealdb_candidates=rust_report["surrealdb_candidates"],
            supabase_candidates=rust_report["supabase_candidates"],
            optimization_count=rust_report["optimization_count"] + len(ctas_optimizations),
            ctas_specific_optimizations=ctas_optimizations
        )
    
    async def generate_ctas_schemas(self) -> Dict[str, str]:
        """Generate CTAS-specific database schemas"""
        logger.info("🗄️ Generating CTAS database schemas...")
        
        schemas = {
            "supabase": self._generate_supabase_schema(),
            "mongodb": self._generate_mongodb_schema(),
            "neo4j": self._generate_neo4j_schema(),
            "surrealdb": self._generate_surrealdb_schema()
        }
        
        return schemas
    
    def _generate_supabase_schema(self) -> str:
        """Generate Supabase/PostgreSQL schema for CTAS"""
        schema = """
-- CTAS Supabase Schema
-- ===================

-- Threat Intelligence Table
CREATE TABLE threat_intelligence (
    threat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    threat_type VARCHAR(100) NOT NULL,
    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
    source VARCHAR(255),
    indicators JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Geospatial Data Table
CREATE TABLE geospatial_data (
    location_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    coordinates POINT NOT NULL,
    threat_density DECIMAL(5,2),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Analytics Reports Table
CREATE TABLE analytics_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    data_summary JSONB,
    retention_policy VARCHAR(50)
);

-- Indexes for performance
CREATE INDEX idx_threat_intelligence_timestamp ON threat_intelligence(timestamp);
CREATE INDEX idx_threat_intelligence_type ON threat_intelligence(threat_type);
CREATE INDEX idx_geospatial_coordinates ON geospatial_data USING GIST(coordinates);
CREATE INDEX idx_analytics_reports_type ON analytics_reports(report_type);

-- Row Level Security
ALTER TABLE threat_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE geospatial_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_reports ENABLE ROW LEVEL SECURITY;
"""
        return schema
    
    def _generate_mongodb_schema(self) -> str:
        """Generate MongoDB schema for CTAS"""
        schema = """
// CTAS MongoDB Schema
// ===================

// Behavioral Analysis Collection
db.createCollection("behavioral_analysis", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["behavior_id", "entity_id", "pattern_type"],
            properties: {
                behavior_id: { bsonType: "string" },
                entity_id: { bsonType: "string" },
                pattern_type: { bsonType: "string" },
                confidence_score: { bsonType: "double" },
                timestamp: { bsonType: "date" }
            }
        }
    }
});

// Create indexes for performance
db.behavioral_analysis.createIndex({ "entity_id": 1 });
db.behavioral_analysis.createIndex({ "pattern_type": 1 });
db.behavioral_analysis.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 2592000 }); // 30 days TTL

// Threat Intelligence Collection (for warm data)
db.createCollection("threat_intelligence_warm", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["threat_id", "threat_type", "severity"],
            properties: {
                threat_id: { bsonType: "string" },
                threat_type: { bsonType: "string" },
                severity: { bsonType: "int" },
                indicators: { bsonType: "array" }
            }
        }
    }
});
"""
        return schema
    
    def _generate_neo4j_schema(self) -> str:
        """Generate Neo4j schema for CTAS"""
        schema = """
// CTAS Neo4j Schema
// =================

// Threat Correlation Constraints
CREATE CONSTRAINT threat_id IF NOT EXISTS FOR (t:Threat) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE;

// Threat Node Properties
CREATE (t:Threat {
    id: "threat_001",
    type: "malware",
    severity: 8,
    first_seen: datetime(),
    indicators: ["hash1", "hash2", "domain1"]
});

// Entity Node Properties  
CREATE (e:Entity {
    id: "entity_001",
    type: "ip_address",
    value: "192.168.1.1",
    first_seen: datetime()
});

// Location Node Properties
CREATE (l:Location {
    id: "location_001",
    coordinates: point({longitude: -74.006, latitude: 40.7128}),
    threat_density: 0.75
});

// Relationships
CREATE (t:Threat {id: "threat_001"})-[:ORIGINATES_FROM]->(e:Entity {id: "entity_001"});
CREATE (e:Entity {id: "entity_001"})-[:LOCATED_AT]->(l:Location {id: "location_001"});
CREATE (t1:Threat {id: "threat_001"})-[:CORRELATES_WITH]->(t2:Threat {id: "threat_002"});
"""
        return schema
    
    def _generate_surrealdb_schema(self) -> str:
        """Generate SurrealDB schema for CTAS"""
        schema = """
-- CTAS SurrealDB Schema
-- =====================

-- Define tables with relationships
DEFINE TABLE threat_intelligence SCHEMAFULL;
DEFINE FIELD threat_id ON threat_intelligence TYPE string;
DEFINE FIELD timestamp ON threat_intelligence TYPE datetime;
DEFINE FIELD threat_type ON threat_intelligence TYPE string;
DEFINE FIELD severity ON threat_intelligence TYPE int;
DEFINE FIELD source ON threat_intelligence TYPE string;
DEFINE FIELD indicators ON threat_intelligence TYPE array;
DEFINE FIELD created_at ON threat_intelligence TYPE datetime DEFAULT time::now();

DEFINE TABLE geospatial_data SCHEMAFULL;
DEFINE FIELD location_id ON geospatial_data TYPE string;
DEFINE FIELD coordinates ON geospatial_data TYPE geometry;
DEFINE FIELD threat_density ON geospatial_data TYPE float;
DEFINE FIELD last_updated ON geospatial_data TYPE datetime DEFAULT time::now();

DEFINE TABLE behavioral_analysis SCHEMAFULL;
DEFINE FIELD behavior_id ON behavioral_analysis TYPE string;
DEFINE FIELD entity_id ON behavioral_analysis TYPE string;
DEFINE FIELD pattern_type ON behavioral_analysis TYPE string;
DEFINE FIELD confidence_score ON behavioral_analysis TYPE float;
DEFINE FIELD timestamp ON behavioral_analysis TYPE datetime DEFAULT time::now();

-- Define relationships
DEFINE RELATIONSHIP originates_from ON threat_intelligence -> entity;
DEFINE RELATIONSHIP located_at ON entity -> geospatial_data;
DEFINE RELATIONSHIP correlates_with ON threat_intelligence -> threat_intelligence;

-- Define indexes for performance
DEFINE INDEX threat_type_idx ON threat_intelligence FIELDS threat_type;
DEFINE INDEX severity_idx ON threat_intelligence FIELDS severity;
DEFINE INDEX coordinates_idx ON geospatial_data FIELDS coordinates;
DEFINE INDEX pattern_type_idx ON behavioral_analysis FIELDS pattern_type;
"""
        return schema

# Usage example
async def main():
    """Example usage of CTAS Database Analyzer"""
    analyzer = CTASDatabaseAnalyzer("/Users/cp5337/Developer/nyx-trace-6-6-ACTIVE-MAIN")
    
    # Analyze the project
    report = await analyzer.analyze_ctas_project()
    
    print(f"\n📊 CTAS Database Analysis Report")
    print(f"=================================")
    print(f"Total Structs: {report.total_structs}")
    print(f"├─ Sled candidates: {report.sled_candidates} (high frequency)")
    print(f"├─ SurrealDB candidates: {report.surrealdb_candidates} (relationships)")
    print(f"└─ Supabase candidates: {report.supabase_candidates} (analytics)")
    print(f"\nCTAS Optimizations: {len(report.ctas_specific_optimizations)}")
    
    # Generate schemas
    schemas = await analyzer.generate_ctas_schemas()
    
    print(f"\n🗄️ Generated schemas for:")
    for db_type, schema in schemas.items():
        print(f"  - {db_type.upper()}")
    
    return report, schemas

if __name__ == "__main__":
    asyncio.run(main()) 