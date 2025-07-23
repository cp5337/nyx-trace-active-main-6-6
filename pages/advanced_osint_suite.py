"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PAGES-ADVANCED-OSINT-0001           │
// │ 📁 domain       : UI, OSINT, Intelligence                   │
// │ 🧠 description  : Advanced OSINT Suite providing integrated │
// │                  access to multiple intelligence sources     │
// │ 🕸️ hash_type    : UUID → CUID-linked interface              │
// │ 🔄 parent_node  : NODE_INTERFACE                           │
// │ 🧩 dependencies : streamlit, core modules                   │
// │ 🔧 tool_usage   : Analysis, Collection, Intelligence        │
// │ 📡 input_type   : URLs, media, search terms                 │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : information gathering, analysis           │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Advanced OSINT Suite
------------------
Comprehensive OSINT (Open Source Intelligence) Suite providing integrated
access to web intelligence, media analysis, and darkweb monitoring tools
with advanced analysis capabilities for professional intelligence gathering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import time
import logging
from datetime import datetime, timedelta
from PIL import Image
import io
import hashlib

from core.web_intelligence.news_scraper import NewsScraper, NewsArticle
from core.media_analysis.analyzer import MediaAnalyzer
from core.darkweb_analyzer.darkweb_intelligence import DarkwebIntelligenceEngine, DarkwebMonitoringTarget

# Configure logging
logger = logging.getLogger("ctas_osint_suite")
logger.setLevel(logging.INFO)

# Function configures subject page
# Method sets predicate properties
# Operation defines object settings
def configure_page():
    """
    Configure the Streamlit page settings
    
    # Function configures subject page
    # Method sets predicate properties
    # Operation defines object settings
    """
    st.set_page_config(
        page_title="Advanced OSINT Suite - NyxTrace",
        page_icon=None,  # Professional approach without icons
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for professional appearance
    st.markdown("""
    <style>
    .data-frame {
        font-family: "IBM Plex Mono", monospace;
        font-size: 12px;
    }
    .result-card {
        background-color: #f9f9f9;
        border-left: 3px solid #2c3e50;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .url-display {
        font-family: "IBM Plex Mono", monospace;
        font-size: 14px;
        word-break: break-all;
        background-color: #f5f5f5;
        padding: 5px;
        border-radius: 3px;
    }
    .metadata-table {
        font-size: 14px;
    }
    .metadata-table td {
        padding: 3px 10px 3px 0;
    }
    .metadata-table th {
        font-weight: bold;
        padding: 3px 10px 3px 0;
        text-align: left;
    }
    .analysis-section {
        margin: 15px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }
    .section-header {
        color: #2c3e50;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# Function initializes subject components
# Method creates predicate instances
# Operation prepares object analyzers
def initialize_components():
    """
    Initialize all analysis components required for the OSINT suite
    
    # Function initializes subject components
    # Method creates predicate instances
    # Operation prepares object analyzers
    """
    # Create data directories if they don't exist
    os.makedirs("data/news_cache", exist_ok=True)
    os.makedirs("data/news_sources", exist_ok=True)
    os.makedirs("data/media_analysis/cache", exist_ok=True)
    os.makedirs("data/darkweb", exist_ok=True)
    os.makedirs("data/darkweb_cache", exist_ok=True)
    
    # Initialize components if not already in session state
    if "news_scraper" not in st.session_state:
        st.session_state.news_scraper = NewsScraper(
            cache_dir="data/news_cache",
            news_db_path="data/news_sources/news_organizations.csv"
        )
    
    if "media_analyzer" not in st.session_state:
        st.session_state.media_analyzer = MediaAnalyzer(
            cache_dir="data/media_analysis/cache"
        )
    
    if "darkweb_analyzer" not in st.session_state:
        st.session_state.darkweb_analyzer = DarkwebIntelligenceEngine(
            cache_dir="data/darkweb_cache",
            db_path="data/darkweb/darkweb_intelligence.db"
        )


# Function renders subject header
# Method displays predicate title
# Operation shows object interface
def render_header():
    """
    Render the OSINT Suite header section
    
    # Function renders subject header
    # Method displays predicate title
    # Operation shows object interface
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("Advanced OSINT Suite")
        st.markdown("Professional OSINT capabilities for comprehensive intelligence gathering and analysis")
    
    with col2:
        # Display current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"Current Time: **{current_time}**")
        
        # Display system status
        st.markdown("**System Status**: Operational")
    
    st.markdown("---")


# Function renders subject interface
# Method displays predicate components
# Operation shows object tools
def render_osint_interface():
    """
    Render the main OSINT Suite interface
    
    # Function renders subject interface
    # Method displays predicate components
    # Operation shows object tools
    """
    # Create tabs for different OSINT capabilities
    web_intel_tab, media_analysis_tab, darkweb_tab = st.tabs([
        "Web Intelligence", 
        "Media Analysis", 
        "Darkweb Intelligence"
    ])
    
    # Web Intelligence Tab
    with web_intel_tab:
        render_web_intelligence_section()
    
    # Media Analysis Tab
    with media_analysis_tab:
        render_media_analysis_section()
    
    # Darkweb Intelligence Tab
    with darkweb_tab:
        render_darkweb_intelligence_section()


# Function renders subject section
# Method displays predicate tools
# Operation shows object features
def render_web_intelligence_section():
    """
    Render the Web Intelligence section
    
    # Function renders subject section
    # Method displays predicate tools
    # Operation shows object features
    """
    st.header("Web Intelligence")
    
    st.write("""
    Extract and analyze content from news sources, websites, and online media.
    Supports content extraction, metadata analysis, and intelligence gathering from publicly available sources.
    """)
    
    # Create two columns for input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Article Extraction")
        url = st.text_input("Enter URL to extract and analyze", key="article_url")
        
        if st.button("Extract Article", key="extract_btn"):
            if url:
                # Show spinner during extraction
                with st.spinner("Extracting article content..."):
                    # Extract article
                    article = st.session_state.news_scraper.extract_article_content(url)
                    
                    if article:
                        display_article_results(article)
                    else:
                        st.error("Failed to extract article content.")
            else:
                st.warning("Please enter a URL to extract.")
    
    with col2:
        st.subheader("News Source Analysis")
        source_url = st.text_input("Enter news source URL to analyze", key="source_url")
        article_count = st.slider("Number of articles to analyze", min_value=3, max_value=20, value=5)
        
        if st.button("Analyze Source", key="analyze_source_btn"):
            if source_url:
                with st.spinner("Analyzing news source..."):
                    # Analyze news source
                    results = st.session_state.news_scraper.analyze_news_source(source_url, article_count=article_count)
                    
                    if results and results["articles_analyzed"] > 0:
                        display_source_analysis(results)
                    else:
                        st.error("Failed to analyze news source or no articles found.")
            else:
                st.warning("Please enter a news source URL to analyze.")


# Function displays subject article
# Method shows predicate content
# Operation presents object analysis
def display_article_results(article: NewsArticle):
    """
    Display extracted article results
    
    # Function displays subject article
    # Method shows predicate content
    # Operation presents object analysis
    
    Args:
        article: Extracted NewsArticle object
    """
    st.success("Article extracted successfully")
    
    # Display article title and source
    st.markdown(f"## {article.title}")
    st.markdown(f"**Source**: {article.source_name or article.domain}")
    
    # Display publication date if available
    if article.date_published:
        st.markdown(f"**Published**: {article.date_published.strftime('%Y-%m-%d %H:%M')}")
    
    # Create columns for metadata and content
    meta_col, content_col = st.columns([1, 2])
    
    with meta_col:
        st.subheader("Metadata")
        
        metadata_html = f"""
        <table class="metadata-table">
            <tr><th>Domain</th><td>{article.domain}</td></tr>
            <tr><th>Language</th><td>{article.language or 'Unknown'}</td></tr>
            <tr><th>Word Count</th><td>{article.word_count}</td></tr>
            <tr><th>Author</th><td>{article.author or 'Unknown'}</td></tr>
        </table>
        """
        st.markdown(metadata_html, unsafe_allow_html=True)
        
        # Display categories/tags if available
        if article.categories:
            st.markdown("**Categories/Tags**:")
            st.markdown(", ".join(article.categories))
        
        # Display main image if available
        if article.main_image_url:
            try:
                st.subheader("Main Image")
                st.image(article.main_image_url)
            except:
                st.warning("Failed to load main image")
    
    with content_col:
        st.subheader("Content")
        
        # Display the article text
        if article.text:
            st.markdown(article.text)
        else:
            st.warning("No article text was extracted")
    
    # Display all found images
    if len(article.image_urls) > 1:  # If there are additional images beyond the main one
        st.subheader("All Images")
        
        # Create a grid for images
        image_cols = st.columns(min(4, len(article.image_urls)))
        
        for i, img_url in enumerate(article.image_urls):
            if img_url != article.main_image_url:  # Skip main image as it's already displayed
                with image_cols[i % 4]:
                    try:
                        st.image(img_url, width=150)
                    except:
                        st.write("⚠️ Failed to load image")


# Function displays subject analysis
# Method shows predicate results
# Operation presents object findings
def display_source_analysis(results: dict):
    """
    Display news source analysis results
    
    # Function displays subject analysis
    # Method shows predicate results
    # Operation presents object findings
    
    Args:
        results: News source analysis results dictionary
    """
    st.success("News source analyzed successfully")
    
    # Display source information
    st.subheader(f"Analysis of {results['domain']}")
    
    # Create columns for stats and articles
    stats_col, articles_col = st.columns([1, 2])
    
    with stats_col:
        st.markdown("### Statistics")
        
        stats_html = f"""
        <table class="metadata-table">
            <tr><th>Articles Analyzed</th><td>{results['articles_analyzed']}</td></tr>
            <tr><th>Total Words</th><td>{results['total_words']}</td></tr>
            <tr><th>Avg. Article Length</th><td>{int(results['avg_article_length'])} words</td></tr>
            <tr><th>Primary Language</th><td>{results['language'] or 'Unknown'}</td></tr>
        </table>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
        
        # Display common categories if available
        if results['common_categories']:
            st.markdown("### Common Categories")
            for category in results['common_categories']:
                st.markdown(f"- {category}")
    
    with articles_col:
        st.markdown("### Analyzed Articles")
        
        # Display list of analyzed articles
        if results['article_urls']:
            for i, url in enumerate(results['article_urls'], 1):
                st.markdown(f"{i}. [{url.split('//')[-1].split('/', 1)[0]}]({url})")
        else:
            st.info("No articles were analyzed")


# Function renders subject section
# Method displays predicate tools
# Operation shows object features
def render_media_analysis_section():
    """
    Render the Media Analysis section
    
    # Function renders subject section
    # Method displays predicate tools
    # Operation shows object features
    """
    st.header("Media Analysis")
    
    st.write("""
    Analyze images and videos for intelligence gathering. Extract metadata, detect faces,
    identify objects, and perform comprehensive media analysis for OSINT purposes.
    """)
    
    # Create tabs for different media analysis modes
    url_tab, upload_tab = st.tabs(["Analyze Media from URL", "Upload Media for Analysis"])
    
    # URL Analysis Tab
    with url_tab:
        st.subheader("Analyze Media from URL")
        
        media_url = st.text_input("Enter URL to media file (image or video)", key="media_url")
        media_type = st.selectbox("Select media type", ["Auto-detect", "Image", "Video"])
        
        analyze_options = st.expander("Analysis Options")
        with analyze_options:
            detect_faces = st.checkbox("Detect Faces", value=True)
            detect_objects = st.checkbox("Detect Objects", value=True)
            extract_text = st.checkbox("Extract Text (OCR)", value=False)
        
        if st.button("Analyze Media", key="analyze_media_url_btn"):
            if media_url:
                with st.spinner("Analyzing media..."):
                    # Convert selected media type to format expected by analyzer
                    selected_type = None
                    if media_type == "Image":
                        selected_type = "image"
                    elif media_type == "Video":
                        selected_type = "video"
                    
                    # Analyze media URL
                    results = st.session_state.media_analyzer.analyze_url_media(
                        media_url, 
                        media_type=selected_type
                    )
                    
                    if results and 'error' not in results:
                        display_media_analysis_results(results)
                    else:
                        err_msg = results.get('error', 'Failed to analyze media')
                        st.error(f"Analysis failed: {err_msg}")
            else:
                st.warning("Please enter a media URL to analyze.")
    
    # Upload Analysis Tab
    with upload_tab:
        st.subheader("Upload Media for Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload an image or video file", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'mp4', 'mov']
        )
        
        analyze_options = st.expander("Analysis Options")
        with analyze_options:
            detect_faces_upload = st.checkbox("Detect Faces", value=True, key="faces_upload")
            detect_objects_upload = st.checkbox("Detect Objects", value=True, key="objects_upload")
            extract_text_upload = st.checkbox("Extract Text (OCR)", value=False, key="text_upload")
        
        if uploaded_file is not None:
            # Create a temporary file to analyze
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            
            # Display file info
            st.write(f"File: **{file_details['FileName']}**")
            
            # Check if it's an image or video
            if 'image' in file_details['FileType']:
                # Display preview
                st.image(uploaded_file, caption="Uploaded Image", width=400)
                
                if st.button("Analyze Uploaded Image", key="analyze_upload_btn"):
                    with st.spinner("Analyzing image..."):
                        # Save the uploaded file temporarily
                        temp_file_path = f"data/media_analysis/temp_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Analyze the image
                        results = st.session_state.media_analyzer.analyze_image(
                            temp_file_path,
                            detect_objects=detect_objects_upload,
                            detect_faces=detect_faces_upload,
                            ocr_text=extract_text_upload
                        )
                        
                        # Clean up temp file
                        os.remove(temp_file_path)
                        
                        # Display results
                        display_media_analysis_results({'metadata': results['metadata'], **results})
            
            elif 'video' in file_details['FileType']:
                st.video(uploaded_file)
                
                if st.button("Analyze Uploaded Video", key="analyze_video_btn"):
                    st.warning("Video analysis is resource-intensive and may take some time.")
                    with st.spinner("Analyzing video..."):
                        # Save the uploaded file temporarily
                        temp_file_path = f"data/media_analysis/temp_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Analyze the video
                        results = st.session_state.media_analyzer.analyze_video(
                            temp_file_path,
                            sample_rate=15  # Sample every 15 frames to save time
                        )
                        
                        # Clean up temp file
                        os.remove(temp_file_path)
                        
                        # Display results
                        display_media_analysis_results(results)
            
            else:
                st.error("Unsupported file type. Please upload an image or video file.")


# Function displays subject results
# Method shows predicate analysis
# Operation presents object findings
def display_media_analysis_results(results: dict):
    """
    Display media analysis results
    
    # Function displays subject results
    # Method shows predicate analysis
    # Operation presents object findings
    
    Args:
        results: Media analysis results dictionary
    """
    st.success("Media analysis completed successfully")
    
    # Display metadata
    if 'metadata' in results:
        metadata = results['metadata']
        
        st.subheader("Media Information")
        
        # Create columns for basic info and technical details
        info_col, tech_col = st.columns(2)
        
        with info_col:
            st.markdown("### Basic Information")
            
            info_html = f"""
            <table class="metadata-table">
                <tr><th>Type</th><td>{metadata.file_type.capitalize()}</td></tr>
                <tr><th>Size</th><td>{metadata.size_bytes / 1024:.1f} KB</td></tr>
            """
            
            if metadata.dimensions:
                info_html += f"<tr><th>Dimensions</th><td>{metadata.dimensions[0]} × {metadata.dimensions[1]}</td></tr>"
            
            if metadata.duration:
                info_html += f"<tr><th>Duration</th><td>{metadata.duration:.2f} seconds</td></tr>"
                
            info_html += "</table>"
            st.markdown(info_html, unsafe_allow_html=True)
        
        with tech_col:
            st.markdown("### Technical Details")
            
            tech_html = f"""
            <table class="metadata-table">
                <tr><th>File Hash</th><td><code>{metadata.file_hash[:16]}...</code></td></tr>
                <tr><th>Modified Date</th><td>{metadata.modified_date.strftime('%Y-%m-%d %H:%M:%S') if metadata.modified_date else 'Unknown'}</td></tr>
            """
            
            if metadata.location_data:
                lat = metadata.location_data.get('latitude')
                lon = metadata.location_data.get('longitude')
                tech_html += f"<tr><th>Location Data</th><td>Lat: {lat}, Long: {lon}</td></tr>"
            
            if metadata.device_info:
                device_info = []
                for key, value in metadata.device_info.items():
                    device_info.append(f"{key}: {value}")
                tech_html += f"<tr><th>Device Info</th><td>{'<br>'.join(device_info)}</td></tr>"
                
            tech_html += "</table>"
            st.markdown(tech_html, unsafe_allow_html=True)
    
    # Display content analysis results based on media type
    if 'metadata' in results and results['metadata'].file_type == 'image':
        # Display image analysis results
        display_image_analysis_results(results)
    elif 'metadata' in results and results['metadata'].file_type == 'video':
        # Display video analysis results
        display_video_analysis_results(results)
    else:
        # Display generic results
        st.json(results)


# Function displays subject results
# Method shows predicate analysis
# Operation presents object findings
def display_image_analysis_results(results: dict):
    """
    Display image-specific analysis results
    
    # Function displays subject results
    # Method shows predicate analysis
    # Operation presents object findings
    
    Args:
        results: Image analysis results dictionary
    """
    # Display dominant colors if available
    if 'dominant_colors' in results and results['dominant_colors']:
        st.subheader("Dominant Colors")
        
        # Create color swatches
        color_cols = st.columns(min(5, len(results['dominant_colors'])))
        
        for i, color_info in enumerate(results['dominant_colors']):
            with color_cols[i]:
                hex_color = color_info['hex']
                percentage = color_info['percentage'] * 100
                
                # Use HTML to display color swatch
                st.markdown(f"""
                <div style="background-color: {hex_color}; height: 50px; border-radius: 3px;"></div>
                <div style="text-align: center; margin-top: 5px;">
                    <code>{hex_color}</code><br>
                    {percentage:.1f}%
                </div>
                """, unsafe_allow_html=True)
    
    # Display face detection results if available
    if 'faces' in results and results['faces']:
        st.subheader(f"Detected Faces ({len(results['faces'])})")
        
        # In a real application, you would display the image with bounding boxes
        for i, face in enumerate(results['faces']):
            st.markdown(f"""
            <div class="result-card">
                <h4>Face #{i+1}</h4>
                <p>Confidence: {face['confidence'] * 100:.1f}%</p>
                <p>Bounding Box: {face['bounding_box']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display object detection results if available
    if 'objects' in results and results['objects']:
        st.subheader(f"Detected Objects ({len(results['objects'])})")
        
        for i, obj in enumerate(results['objects']):
            st.markdown(f"""
            <div class="result-card">
                <h4>{obj.get('class', f'Object #{i+1}')}</h4>
                <p>Confidence: {obj.get('confidence', 0) * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display extracted text if available
    if 'text' in results and results['text']:
        st.subheader("Extracted Text (OCR)")
        st.text(results['text'])


# Function displays subject results
# Method shows predicate analysis
# Operation presents object findings
def display_video_analysis_results(results: dict):
    """
    Display video-specific analysis results
    
    # Function displays subject results
    # Method shows predicate analysis
    # Operation presents object findings
    
    Args:
        results: Video analysis results dictionary
    """
    # Display video summary
    st.subheader("Video Analysis Summary")
    
    if 'keyframes' in results:
        st.markdown(f"**Analyzed Keyframes**: {len(results['keyframes'])}")
        
        if results['keyframes']:
            # Display timeline of key events
            st.subheader("Video Timeline")
            
            # Create a dataframe for timeline visualization
            timeline_data = []
            
            for frame in results['keyframes']:
                if 'timestamp' in frame:
                    # Format timestamp as minutes:seconds
                    timestamp = frame['timestamp']
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                    
                    # Get number of faces and objects
                    num_faces = len(frame.get('faces', []))
                    num_objects = len(frame.get('objects', []))
                    
                    # Add to timeline data
                    timeline_data.append({
                        'Timestamp': time_str,
                        'Raw Timestamp': timestamp,
                        'Faces': num_faces,
                        'Objects': num_objects
                    })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df.drop(columns=['Raw Timestamp']), use_container_width=True)
                
                # Display sample keyframes
                st.subheader("Sample Keyframes")
                st.markdown("Displaying analysis of selected keyframes from the video:")
                
                # Select a few keyframes to display (first, middle, last)
                keyframes_to_show = []
                
                if len(results['keyframes']) > 0:
                    keyframes_to_show.append(results['keyframes'][0])  # First
                    
                if len(results['keyframes']) > 2:
                    middle_idx = len(results['keyframes']) // 2
                    keyframes_to_show.append(results['keyframes'][middle_idx])  # Middle
                    
                if len(results['keyframes']) > 1:
                    keyframes_to_show.append(results['keyframes'][-1])  # Last
                
                # Display each keyframe's analysis
                for i, keyframe in enumerate(keyframes_to_show):
                    # Format timestamp
                    timestamp = keyframe.get('timestamp', 0)
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                    
                    st.markdown(f"### Keyframe at {time_str}")
                    
                    # Display keyframe analysis (using image analysis display function)
                    with st.expander("View Keyframe Analysis", expanded=False):
                        display_image_analysis_results(keyframe)
    else:
        st.info("No keyframe analysis available for this video.")


# Function renders subject section
# Method displays predicate tools
# Operation shows object features
def render_darkweb_intelligence_section():
    """
    Render the Darkweb Intelligence section
    
    # Function renders subject section
    # Method displays predicate tools
    # Operation shows object features
    """
    st.header("Darkweb Intelligence")
    
    st.write("""
    Access darkweb intelligence gathering and monitoring capabilities for OSINT investigations.
    Monitor onion sites, track keywords, and analyze content in a secure environment.
    """)
    
    # Warning about accessing darkweb content
    st.warning("""
    **Important Notice**: Darkweb access requires proper security configuration including a TOR proxy.
    This tool only analyzes previously collected and cached darkweb intelligence to maintain operational security.
    """)
    
    # Check if TOR proxy is configured
    if not st.session_state.darkweb_analyzer.proxy_url:
        st.error("TOR proxy not configured. Darkweb direct access is disabled.")
    
    # Create tabs for different darkweb analysis functions
    monitor_tab, query_tab, target_tab = st.tabs([
        "Monitoring Dashboard", 
        "Query Cached Intelligence",
        "Monitoring Targets"
    ])
    
    # Monitoring Dashboard Tab
    with monitor_tab:
        st.subheader("Darkweb Monitoring Dashboard")
        
        # Get alerts
        alerts = st.session_state.darkweb_analyzer.get_alerts(
            start_date=datetime.now() - timedelta(days=7)  # Last 7 days
        )
        
        if alerts:
            st.success(f"Found {len(alerts)} alerts from monitoring")
            
            # Display alerts
            for alert in alerts:
                st.markdown(f"""
                <div class="result-card">
                    <h4>{alert.get('title', 'Unnamed Alert')}</h4>
                    <p><strong>URL:</strong> <span class="url-display">{alert.get('url', 'Unknown')}</span></p>
                    <p><strong>Score:</strong> {alert.get('score', 0) * 100:.1f}%</p>
                    <p><strong>Date:</strong> {alert.get('alert_date').strftime('%Y-%m-%d %H:%M:%S') if isinstance(alert.get('alert_date'), datetime) else 'Unknown'}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent monitoring alerts found.")
            
            # Add a demo alert for visualization
            st.markdown("### Example Alert (Demo)")
            st.markdown("""
            <div class="result-card">
                <h4>Suspicious Market Activity Detected</h4>
                <p><strong>URL:</strong> <span class="url-display">abcdefg1234567890.onion</span></p>
                <p><strong>Score:</strong> 87.5%</p>
                <p><strong>Date:</strong> 2025-05-10 14:23:17</p>
                <p><strong>Keywords:</strong> 5 matches found</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Query Intelligence Tab
    with query_tab:
        st.subheader("Query Cached Darkweb Intelligence")
        
        # Search form
        search_term = st.text_input("Search Term", key="darkweb_search")
        
        if st.button("Search", key="darkweb_search_btn"):
            if search_term:
                with st.spinner("Searching cached darkweb intelligence..."):
                    results = st.session_state.darkweb_analyzer.search_darkweb_content(search_term)
                    
                    if results:
                        st.success(f"Found {len(results)} results matching '{search_term}'")
                        
                        # Display results
                        for result in results:
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>{result.get('title', 'Untitled Content')}</h4>
                                <p><strong>URL:</strong> <span class="url-display">{result.get('url', 'Unknown')}</span></p>
                                <p><strong>Site Type:</strong> {result.get('site_type', 'Unknown').capitalize()}</p>
                                <p><strong>Extraction Date:</strong> {result.get('extraction_date') if result.get('extraction_date') else 'Unknown'}</p>
                                
                                {f"<p><strong>Match:</strong><br>{result.get('snippet')}</p>" if result.get('snippet') else ""}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"No results found matching '{search_term}'")
            else:
                st.warning("Please enter a search term")
    
    # Monitoring Targets Tab
    with target_tab:
        st.subheader("Darkweb Monitoring Targets")
        
        # Get existing targets
        targets = st.session_state.darkweb_analyzer.get_monitoring_targets()
        
        if targets:
            st.write(f"Currently monitoring {len(targets)} targets")
            
            # Display targets
            for target in targets:
                status = "Active" if target.is_active else "Inactive"
                status_color = "green" if target.is_active else "red"
                
                st.markdown(f"""
                <div class="result-card">
                    <h4>{target.name} <span style="color: {status_color};">({status})</span></h4>
                    <p><strong>Keywords:</strong> {', '.join(target.keywords)}</p>
                    <p><strong>Frequency:</strong> Every {target.monitoring_frequency} hours</p>
                    <p><strong>URLs Monitored:</strong> {len(target.urls)}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No monitoring targets configured")
            
            # Add demo target for visualization
            st.markdown("### Example Target (Demo)")
            st.markdown("""
            <div class="result-card">
                <h4>Cryptocurrency Monitoring <span style="color: green;">(Active)</span></h4>
                <p><strong>Keywords:</strong> bitcoin, crypto, blockchain, wallet, exchange</p>
                <p><strong>Frequency:</strong> Every 12 hours</p>
                <p><strong>URLs Monitored:</strong> 5</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add new target form
        st.subheader("Add New Monitoring Target")
        with st.expander("New Target Form", expanded=False):
            # Target information form
            target_name = st.text_input("Target Name", key="new_target_name")
            keywords = st.text_input("Keywords (comma-separated)", key="new_target_keywords")
            urls = st.text_area("URLs to Monitor (one per line)", key="new_target_urls")
            frequency = st.slider("Monitoring Frequency (hours)", 1, 72, 24, key="new_target_freq")
            
            if st.button("Add Target", key="add_target_btn"):
                if target_name and keywords:
                    # Create target object
                    new_target = DarkwebMonitoringTarget(
                        name=target_name,
                        keywords=[k.strip() for k in keywords.split(',') if k.strip()],
                        urls=[u.strip() for u in urls.splitlines() if u.strip()],
                        monitoring_frequency=frequency
                    )
                    
                    # Save target
                    success = st.session_state.darkweb_analyzer.add_monitoring_target(new_target)
                    
                    if success:
                        st.success(f"Added monitoring target: {target_name}")
                        st.rerun()  # Refresh the page to show the new target
                    else:
                        st.error("Failed to add monitoring target")
                else:
                    st.warning("Please enter at least a target name and keywords")


# Main function for OSINT Suite
# Function runs subject suite
# Method executes predicate application
# Operation starts object interface
def main():
    """
    Main function to run the OSINT Suite
    
    # Function runs subject suite
    # Method executes predicate application
    # Operation starts object interface
    """
    # Configure page
    configure_page()
    
    # Initialize components
    initialize_components()
    
    # Render header
    render_header()
    
    # Render main interface
    render_osint_interface()


# Run the application
if __name__ == "__main__":
    main()