"""
NyxTrace Download Center
------------------------
Provides access to all handoff package files for download.
"""

import streamlit as st
import os
import base64
from pathlib import Path

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """
    Creates a download link for binary files
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}" class="download-button">{file_label}</a>'
    return href

# Configure page
st.set_page_config(
    page_title="NyxTrace Download Center",
    page_icon="🧩",
    layout="wide",
)

# Apply custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 80rem;
    }
    
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4D96FF;
        color: white !important;
        text-decoration: none;
        border-radius: 4px;
        font-weight: 500;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        transition: background-color 0.3s;
    }
    
    .download-button:hover {
        background-color: #3a7fd5;
        color: white !important;
    }
    
    .file-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #333;
    }
    
    .file-name {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .file-size {
        font-size: 0.8rem;
        color: #aaa;
        margin-bottom: 0.5rem;
    }
    
    .file-description {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #ddd;
    }
    
    /* Fix white space issues */
    h1, h2, h3, p {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #1A1C24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4D96FF !important;
        color: white !important;
    }
    
    /* Remove extra white space */
    .css-1544g2n {
        padding-top: 2rem;
    }
    
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    
    /* Reduce white space in error messages */
    .element-container .stAlert {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling to reduce whitespace
st.markdown('<h1 style="margin-bottom:0.5rem;color:white;">NyxTrace Handoff Package Download Center</h1>', unsafe_allow_html=True)

st.markdown("""
<p style="margin-bottom:1rem;">This page provides access to all components of the NyxTrace handoff package.
Select a category below to view and download the available files.</p>
""", unsafe_allow_html=True)

# Create tabs for different download categories
tab1, tab2, tab3, tab4 = st.tabs(["Complete Package", "Component Archives", "Documentation", "Additional Resources"])

with tab1:
    st.header("Complete Handoff Package")
    st.markdown("This package contains all components and documentation in a single ZIP file.")
    
    complete_package = Path("nyxtrace_complete_handoff_package.zip")
    if complete_package.exists():
        file_size = round(complete_package.stat().st_size / (1024 * 1024), 2)
        
        st.markdown(f"""
        <div class="file-card">
            <div class="file-name">Complete Handoff Package</div>
            <div class="file-size">{file_size} MB</div>
            <div class="file-description">
                Contains all component archives, documentation files, and additional resources in a single zip file.
            </div>
            {get_binary_file_downloader_html(str(complete_package), 'Download Complete Package')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Complete package file not found. Please check the file path.")

with tab2:
    st.header("Component Archives")
    st.markdown("Individual component archives organized by subsystem.")
    
    components = [
        ("Database Layer", "nyxtrace_part1_database.zip", "Contains the database layer components including models, connectors, and query utilities."),
        ("Core Components", "nyxtrace_part2_core.zip", "Contains core system components including the registry and main processing logic."),
        ("Utilities", "nyxtrace_part3_utils.zip", "Contains utility functions and helper modules used throughout the system."),
        ("Documentation", "nyxtrace_part4_docs.zip", "Contains in-depth documentation for all system components."),
        ("Streamlit Frontend", "nyxtrace_streamlit_frontend.zip", "Contains Streamlit frontend components and UI elements."),
        ("Streamlit Pages", "nyxtrace_streamlit_pages.zip", "Contains Streamlit page implementations for the application.")
    ]
    
    for name, filename, description in components:
        filepath = Path(filename)
        if filepath.exists():
            file_size = round(filepath.stat().st_size / 1024, 2)
            
            st.markdown(f"""
            <div class="file-card">
                <div class="file-name">{name}</div>
                <div class="file-size">{file_size} KB</div>
                <div class="file-description">{description}</div>
                {get_binary_file_downloader_html(str(filepath), f'Download {name}')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"File {filename} not found. Please check the file path.")
            
with tab3:
    st.header("Documentation")
    st.markdown("Individual documentation files for the handoff package.")
    
    docs_path = Path("handoff_package")
    if docs_path.exists():
        docs = [
            ("Project Manifest", "00_PROJECT_MANIFEST.md", "Overview of the entire project and its components."),
            ("Refactoring Roadmap", "REFACTORING_ROADMAP.md", "Detailed plan for refactoring the codebase."),
            ("React Migration Guide", "REACT_MIGRATION_GUIDE.md", "Guide for migrating from Streamlit to React."),
            ("Dioxus Migration Guide", "DIOXUS_MIGRATION_GUIDE.md", "Guide for migrating from Streamlit to Dioxus (Rust)."),
            ("Implementation Checklist", "IMPLEMENTATION_CHECKLIST.md", "Checklist of implementation tasks."),
            ("Handoff Summary", "HANDOFF_SUMMARY.md", "Summary of the handoff package contents."),
            ("Screenshot Capture Guide", "SCREENSHOT_CAPTURE_GUIDE.md", "Guide for capturing UI screenshots."),
            ("Handoff Instructions", "HANDOFF_INSTRUCTIONS.md", "Instructions for the handoff process."),
            ("Environment Example", ".env.example", "Example environment configuration file."),
            ("README Template", "README_TEMPLATE.md", "Template for component README files."),
            ("Package Contents", "PACKAGE_CONTENTS.md", "List of all files in the handoff package.")
        ]
        
        for name, filename, description in docs:
            filepath = docs_path / filename
            if filepath.exists():
                file_size = round(filepath.stat().st_size / 1024, 2)
                
                st.markdown(f"""
                <div class="file-card">
                    <div class="file-name">{name}</div>
                    <div class="file-size">{file_size} KB</div>
                    <div class="file-description">{description}</div>
                    {get_binary_file_downloader_html(str(filepath), f'Download {name}')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"File {filename} not found. Please check the file path.")
    else:
        st.error("Documentation directory not found. Please check the path.")
            
with tab4:
    st.header("Additional Resources")
    st.markdown("Additional project resources and backup files.")
    
    resources = [
        ("Full Project Backup", "nyxtrace_project.zip", "Complete backup of the entire project."),
        ("Database Refactor", "nyxtrace_database_refactor.zip", "Files related to database refactoring."),
        ("Backup Files", "nyxtrace_backup_files.zip", "Additional backup files for the project.")
    ]
    
    for name, filename, description in resources:
        filepath = Path(filename)
        if filepath.exists():
            file_size = round(filepath.stat().st_size / (1024 * 1024), 2)
            
            st.markdown(f"""
            <div class="file-card">
                <div class="file-name">{name}</div>
                <div class="file-size">{file_size} MB</div>
                <div class="file-description">{description}</div>
                {get_binary_file_downloader_html(str(filepath), f'Download {name}')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"File {filename} not found. Please check the file path.")

st.markdown("---")
st.markdown("""
**Note:** For large files, the download may take some time to prepare. 
If you encounter any issues, you can access these files directly in the project file system.
""")