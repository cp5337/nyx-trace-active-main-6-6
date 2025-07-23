"""
Download Page
------------
This page provides download links for project files.
"""

import streamlit as st
import base64
import os

# Set page config
st.set_page_config(
    page_title="NyxTrace - Downloads",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("NyxTrace Download Center")
    
    st.markdown("""
    This page provides download links for project files and resources.
    """)
    
    # Project ZIP Download
    st.header("Project Files")
    
    # Check if project zip exists
    zip_path = "nyxtrace_project.zip"
    if os.path.exists(zip_path):
        st.markdown("### Project ZIP")
        st.markdown(f"Download the entire project as a ZIP file (size: {os.path.getsize(zip_path)/1024/1024:.2f} MB)")
        st.markdown(get_binary_file_downloader_html(zip_path, "NyxTrace Project ZIP"), unsafe_allow_html=True)
    else:
        st.error("Project ZIP file not found. Please generate it first.")
    
    # Documentation Downloads
    st.header("Documentation")
    
    docs_files = [
        ("NyxTrace_Application_Overview.md", "Application Overview"),
        ("README.md", "README")
    ]
    
    for file_path, label in docs_files:
        if os.path.exists(file_path):
            st.markdown(f"### {label}")
            with open(file_path, 'r') as f:
                file_content = f.read()
            st.markdown(file_content)
            st.markdown(get_binary_file_downloader_html(file_path, label), unsafe_allow_html=True)

if __name__ == "__main__":
    main()