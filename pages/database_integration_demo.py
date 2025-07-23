"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DEMO-DATABASE-INTEGRATION-0001      │
// │ 📁 domain       : Demonstration, Database                   │
// │ 🧠 description  : Database integration demonstration page   │
// │                  for multi-database architecture            │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DEMONSTRATION                       │
// │ 🧩 dependencies : streamlit, database.utils                │
// │ 🔧 tool_usage   : Demonstration, UI                        │
// │ 📡 input_type   : User interaction                         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : demonstration, visualization              │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Database Integration Demonstration
-------------------------------
This page demonstrates the multi-database integration architecture
of NyxTrace, showing how data can be stored and retrieved from
Supabase (PostgreSQL), Neo4j, and MongoDB.
"""

import streamlit as st
import json
import os
import uuid
from datetime import datetime
import pandas as pd

# Import database utilities
from core.database.config import DatabaseType
from core.database.utils import get_database_manager

# Set page configuration
st.set_page_config(
    page_title="NyxTrace Database Integration",
    page_icon="🔍",
    layout="wide"
)

# Function initializes subject page
# Method configures predicate layout
# Operation defines object structure
def initialize_page():
    """Initialize the page layout and structure"""
    st.title("Multi-Database Integration")
    st.markdown("""
    This dashboard demonstrates NyxTrace's advanced multi-database architecture, 
    integrating Supabase (PostgreSQL), Neo4j, and MongoDB for comprehensive data storage.
    """)

# Function checks subject status
# Method verifies predicate availability 
# Operation confirms object readiness
def check_database_status():
    """Check status of all database connections"""
    db_manager = get_database_manager()
    available_dbs = db_manager.get_available_databases()
    
    st.subheader("Database Availability")
    
    cols = st.columns(3)
    with cols[0]:
        supabase_status = DatabaseType.SUPABASE in available_dbs
        st.metric(
            "Supabase Status", 
            "Connected" if supabase_status else "Disconnected", 
            delta="OK" if supabase_status else "Error"
        )
        
    with cols[1]:
        neo4j_status = DatabaseType.NEO4J in available_dbs
        st.metric(
            "Neo4j Status", 
            "Connected" if neo4j_status else "Disconnected", 
            delta="OK" if neo4j_status else "Error"
        )
        
    with cols[2]:
        mongodb_status = DatabaseType.MONGODB in available_dbs
        st.metric(
            "MongoDB Status", 
            "Connected" if mongodb_status else "Disconnected", 
            delta="OK" if mongodb_status else "Error"
        )
    
    return available_dbs

# Function demonstrates subject operations
# Method shows predicate capabilities
# Operation presents object features
def entity_demo(available_dbs):
    """Demonstrate entity operations"""
    st.subheader("Entity Operations")
    
    if not available_dbs:
        st.warning("No databases are connected. Please configure connection settings.")
        st.info("""
        To connect to databases, you need to set the following environment variables:
        
        **Supabase:**
        - `SUPABASE_URL`: Your Supabase project URL
        - `SUPABASE_KEY`: Your Supabase API key
        - `DATABASE_URL`: PostgreSQL connection string
        
        **Neo4j:**
        - `NEO4J_URI`: Neo4j connection URI
        - `NEO4J_USERNAME`: Neo4j username
        - `NEO4J_PASSWORD`: Neo4j password
        
        **MongoDB:**
        - `MONGODB_URI`: MongoDB connection string
        - `MONGODB_DATABASE`: MongoDB database name
        """)
        return
    
    # Entity creation form
    st.markdown("### Create Entity")
    with st.form("entity_form"):
        entity_name = st.text_input("Entity Name", "Test Entity")
        entity_type = st.selectbox("Entity Type", [
            "Person", "Location", "Organization", "Event", "Intelligence", "Threat"
        ])
        
        cols = st.columns(2)
        with cols[0]:
            description = st.text_area("Description", "Description of the entity")
        with cols[1]:
            attributes = st.text_area("Custom Attributes (JSON format)", '{"priority": "high"}')
        
        # Button to create entity
        submit_button = st.form_submit_button("Create Entity")
        
        if submit_button:
            try:
                # Parse custom attributes
                custom_attrs = json.loads(attributes)
                
                # Create entity data
                entity_data = {
                    "id": str(uuid.uuid4()),
                    "name": entity_name,
                    "type": entity_type,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    **custom_attrs
                }
                
                # Store entity
                db_manager = get_database_manager()
                results = db_manager.store_entity(entity_type.lower(), entity_data)
                
                # Display results
                st.success(f"Entity created with ID: {entity_data['id']}")
                st.json(results)
                
                # Store entity ID in session state for retrieval demo
                if 'created_entities' not in st.session_state:
                    st.session_state.created_entities = []
                
                st.session_state.created_entities.append({
                    "id": entity_data['id'],
                    "name": entity_name,
                    "type": entity_type.lower()
                })
                
            except Exception as e:
                st.error(f"Error creating entity: {str(e)}")
    
    # Entity retrieval demo
    if 'created_entities' in st.session_state and st.session_state.created_entities:
        st.markdown("### Retrieve Entity")
        
        entity_options = [f"{e['name']} ({e['id']})" for e in st.session_state.created_entities]
        selected_entity = st.selectbox("Select Entity", entity_options)
        
        if selected_entity:
            # Extract entity ID from selection
            entity_id = selected_entity.split('(')[-1].strip(')')
            entity_type = next((e['type'] for e in st.session_state.created_entities if e['id'] == entity_id), None)
            
            if entity_id and entity_type:
                # Fetch entity from database
                db_manager = get_database_manager()
                entity = db_manager.get_entity(entity_type, entity_id)
                
                if entity:
                    st.subheader(f"Entity: {entity.get('name', 'Unknown')}")
                    
                    # Convert entity to DataFrame for display
                    if isinstance(entity, dict):
                        # Filter out some fields for cleaner display
                        display_entity = {k: v for k, v in entity.items() if k not in ['_id']}
                        entity_df = pd.DataFrame([display_entity])
                        st.dataframe(entity_df)
                    else:
                        st.json(entity)
                else:
                    st.warning("Entity not found in any database.")
    else:
        st.info("Create an entity first to enable retrieval.")

# Function initializes subject demo
# Method prepares predicate content 
# Operation displays object interface
def main():
    """Main function for the database integration demo page"""
    initialize_page()
    
    # Check database connections
    available_dbs = check_database_status()
    
    # Configuration section
    with st.expander("Database Configuration"):
        st.markdown("""
        ### Database Configuration
        
        NyxTrace uses a multi-database architecture with the following components:
        
        1. **Supabase (PostgreSQL)** - Relational data store for structured intelligence data
        2. **Neo4j** - Graph database for relationship analysis and network visualization
        3. **MongoDB** - Document store for unstructured intelligence artifacts
        
        Each database serves a specific purpose in the NyxTrace ecosystem,
        providing specialized capabilities for different types of intelligence data.
        """)
    
    # Display entity demo
    entity_demo(available_dbs)
    
    # Information about architecture
    with st.expander("Architecture Details"):
        st.markdown("""
        ### Multi-Database Architecture
        
        The NyxTrace database integration architecture provides:
        
        - **Cross-database entity synchronization**
        - **Specialized storage optimized for different data types**
        - **Resilience through data redundancy**
        - **Optimized query patterns for different use cases**
        
        This architecture enables advanced analytics capabilities while
        maintaining performance and data integrity across the system.
        """)
        
        # Architecture diagram
        st.image("https://via.placeholder.com/800x400?text=NyxTrace+Database+Architecture", 
                 caption="NyxTrace Multi-Database Architecture")


if __name__ == "__main__":
    main()