# NyxTrace / CTAS Project Handoff Manifest

## Project Overview

NyxTrace is an advanced geospatial intelligence platform functioning as a specialized subsystem within the Convergent Threat Analysis System (CTAS v6.5) framework. The system visualizes adversary tasks through an interactive "periodic table of nodes" concept and provides comprehensive data organization and visualization capabilities.

## Architecture

The system is built with a modular architecture:

1. **Database Layer**: Thread-safe connectors for Supabase with connection pooling
2. **Core Logic**: Task registry, adversary task models, and relationship management
3. **Utilities**: Helper functions, rendering utilities, and shared tools
4. **Frontend**: Streamlit-based UI components with a potential future migration to React
5. **Visualization**: Geospatial intelligence visualization and interactive dashboards

## ZIP Archive Contents

### 01_nyxtrace_database.zip
- Database connectors and connection management
- Thread-safe database factory and connection pooling
- Database schemas and query utilities

### 02_nyxtrace_core.zip
- Core business logic and data models
- Task registry implementation
- Adversary task definitions and relationship models

### 03_nyxtrace_utils.zip
- Utility functions and helpers
- Rendering utilities and formatters
- Shared tools and common functions

### 04_nyxtrace_docs.zip
- Documentation and specifications
- API documentation and usage guides
- Architecture diagrams and design documents

### 05_nyxtrace_streamlit_frontend.zip
- Frontend components and UI elements
- Shared Streamlit components and layouts
- UI state management and navigation

### 06_nyxtrace_streamlit_pages.zip
- Streamlit page implementations
- Interactive dashboards and visualizations
- User interface screens and workflows

### 07_ui_screenshots
- Screenshots of current UI implementation
- Visual reference for component styling and layout
- Sample dashboards and visualization examples

## Component Dependencies

1. **Database ← Core**: Core components depend on database layer for data access
2. **Core ← Utils**: Core logic uses utility functions for operations
3. **Frontend ← Core**: Frontend components consume core data models and logic
4. **Pages ← Frontend**: Pages implement and compose frontend components
5. **All ← Utils**: All components utilize shared utilities as needed

## Known Issues and Refactoring Priorities

1. **Thread-Safety in Database Layer**:
   - Some Streamlit threading issues with database connections
   - Need for improved connection pooling implementation

2. **HTML Rendering in Task Viewer**:
   - Task cards not rendering correctly in Replit environment
   - Raw HTML displaying instead of rendered content

3. **Code Standardization**:
   - Apply 80-character line length standard
   - Limit module size to 30 lines
   - Improve comment density to 15%

4. **Data Type Consistency**:
   - Inconsistency between Element objects and dictionaries
   - Standardization needed for QueryResult objects

5. **React Migration Preparation**:
   - Clear separation of UI and logic for future React port
   - Component-based design approach
   - Proper state management planning

## Environment Setup

1. **Python Requirements**:
   - Python 3.10+
   - Streamlit 1.25+
   - Required packages: plotly, pandas, numpy, etc.

2. **Database Requirements**:
   - Supabase account and connection credentials
   - Optional: Local PostgreSQL for development

3. **Environment Variables**:
   - See .env.example for required configuration
   - API keys and connection strings must be provided