#!/usr/bin/env python3
"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-UTILS-REPO_ANALYZER-0001            │
// │ 📁 domain       : Code Analysis, Intelligence Extraction    │
// │ 🧠 description  : Codebase Analyzer for Intelligence        │
// │                  Extraction and CTAS Node Mapping           │
// │ 🕸️ hash_type    : UUID → CUID-linked intelligence           │
// │ 🔄 parent_node  : NODE_ANALYSIS                            │
// │ 🧩 dependencies : pathlib, json, logging, ast               │
// │ 🔧 tool_usage   : Intelligence Extraction, CTAS Mapping     │
// │ 📡 input_type   : Codebase files, repositories              │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : feature detection, context preservation   │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

NyxTrace Repository Analyzer
---------------------------
Advanced codebase analysis tool that extracts intelligence from source code,
maps features to CTAS nodes, and preserves development context. This component
enables geospatial and threat intelligence extraction from code repositories.

Core capabilities:
- Structural analysis of codebases with filtering
- Intelligence feature extraction based on keyword patterns
- CTAS node mapping with UUID linking
- Context preservation for development continuity
"""

# System imports subject libraries
# Module loads predicate dependencies
# Package defines object functionality
# Code arranges subject components
import json          # For saving analysis results
import csv          # For reading task mapping CSV
import re           # For keyword pattern matching
from pathlib import Path  # For cross-platform path handling
from datetime import datetime  # For timestamps
from typing import Dict, List, Optional, Any, TypedDict, Counter  # Type hints
import logging      # For logging analysis progress
import sys         # For stdout/stderr handling
import zipfile     # For handling zip archives
import shutil      # For file operations
from collections import defaultdict  # For counting
import time        # For timing operations

# System configures subject output
# Logging stores predicate messages
# Configuration affects object behavior
# Module enhances subject functionality
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),  # Log to file
        logging.StreamHandler(sys.stdout)    # Log to console
    ]
)

def print_progress(message: str, end: str = '\n') -> None:
    """
    Print progress message with timestamp and force flush
    
    # Function generates subject message
    # Output includes predicate timestamp
    # Message contains object information
    # Terminal displays subject progress
    """
    # Time creates subject timestamp
    # Function formats predicate string
    # Format includes object components
    # Output displays subject information
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}", end=end, flush=True)
    sys.stdout.flush()  # Force flush to ensure immediate output

def print_step(step: int, total: int, message: str) -> None:
    """
    Print a numbered step with progress indicator
    
    # Function displays subject progress
    # Output formats predicate step
    # Message contains object information
    # Interface shows subject workflow
    """
    # Function calls subject method
    # Steps track predicate progress
    # Message conveys object information
    # Interface shows subject status
    print_progress(f"Step {step}/{total}: {message}")

# Type defines subject structure
# Class implements predicate typing
# Information contains object fields
# Dictionary represents subject data
class StructureInfo(TypedDict):
    """
    Structure analysis results - tracks file counts and types
    
    # Class defines subject structure
    # Analysis creates predicate results
    # Information contains object metrics
    # Dictionary represents subject data
    """
    files: int              # Total number of files
    code_files: int         # Number of code files
    artifact_files: int     # Number of artifact files
    extensions: Dict[str, int]  # Count of files by extension
    artifacts: Dict[str, List[str]]  # Artifact files by category
    last_modified: Optional[float]  # Most recent file modification time

# Type defines subject feature
# Class implements predicate typing
# Information contains object fields
# Dictionary represents subject data
class FeatureInfo(TypedDict):
    """
    Feature analysis results - tracks keyword matches
    
    # Class defines subject feature
    # Analysis creates predicate results
    # Information contains object patterns
    # Dictionary represents subject data
    """
    file: str      # File where keyword was found
    keyword: str   # Keyword that was matched
    category: str  # Category of the keyword
    context: str   # Surrounding code context

# Type defines subject mapping
# Class implements predicate typing
# Node contains object references
# Dictionary represents subject data
class NodeMapping(TypedDict):
    """
    CTAS node mapping results - maps features to CTAS nodes
    
    # Class defines subject mapping
    # Structure creates predicate relationships
    # Node contains object references
    # Dictionary represents subject connections
    """
    feature: FeatureInfo    # The matched feature
    node: str              # CTAS node name
    uuid: str              # Node UUID
    hd4_pillars: List[str] # HD4 pillars associated with node

# Class defines subject analyzer
# Analysis implements predicate functionality
# Codebase contains object intelligence
# Context preserves subject information
class CodebaseAnalyzer:
    """
    Main analyzer class - handles codebase analysis and context preservation
    
    # Class implements subject analyzer
    # Code processes predicate intelligence
    # Repository contains object features
    # Analysis extracts subject patterns
    """
    
    def __init__(self, input_path: str, output_dir: str) -> None:
        """
        Initialize the analyzer with input/output paths and configuration
        
        # Function initializes subject analyzer
        # Constructor sets predicate configuration
        # Parameters contain object paths
        # Class prepares subject environment
        """
        # Path defines subject location
        # Variable stores predicate reference
        # String contains object information
        # Constructor initializes subject attributes
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.workspace_dir = self.output_dir / 'workspace'  # Where files are copied
        self.context_dir = self.output_dir / 'context'     # Where analysis is saved
        
        # Directory creates subject structure
        # Function makes predicate folders
        # Parameter controls object behavior
        # System organizes subject files
        self.output_dir.mkdir(exist_ok=True)
        self.workspace_dir.mkdir(exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)
        
        # Variable defines subject criteria
        # Set contains predicate extensions
        # Collection stores object values
        # Filter selects subject files
        self.code_extensions = {'.py', '.rs', '.ps1', '.h', '.cpp', '.c', '.hpp', '.cs', '.java', '.ts', '.js'}
        self.exclude_dirs = {
            '__pycache__', 'node_modules', '.git', '.svn', 'build', 'dist', 
            'target', 'venv', '.venv', 'env', '.env', 'bin', 'obj',
            '.mypy_cache', '.pytest_cache', '.coverage', '.tox',
            'coverage', 'htmlcov', '.idea', '.vscode'
        }
        
        # Function loads subject mapping
        # Method gets predicate tasks
        # Variable stores object result
        # Class initializes subject data
        self.task_mapping = self._load_task_mapping()
        
        # Dictionary defines subject categories
        # Keywords create predicate patterns
        # Lists contain object values
        # Analysis uses subject configuration
        self.keyword_categories = {
            'OSINT': ['scrape', 'harvest', 'monitor', 'collect', 'gather'],
            'Geospatial': ['geo', 'location', 'coordinate', 'map', 'route'],
            'Persona': ['identity', 'profile', 'persona', 'actor', 'entity'],
            'Threat': ['threat', 'risk', 'vulnerability', 'attack', 'exploit'],
            'Infrastructure': ['infra', 'system', 'network', 'service', 'node']
        }
        
        # Dictionary defines subject categories
        # Extensions create predicate patterns
        # Sets contain object values
        # Analysis uses subject configuration
        self.artifact_categories = {
            'Documents': {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.rtf'},
            'Media': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg', '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'},
            'Archives': {'.zip', '.tar', '.gz', '.rar', '.7z', '.iso'},
            'Executables': {'.exe', '.dll', '.so', '.dylib', '.bin'},
            'Databases': {'.db', '.sqlite', '.sql', '.mdb', '.accdb'},
            'VirtualDisks': {'.vmdk', '.vhd', '.vhdx', '.qcow2'}
        }
    
    def _load_task_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Load task mapping from CSV file
        
        # Function loads subject mapping
        # Method reads predicate file
        # CSV contains object data
        # Dictionary returns subject structure
        
        Returns:
            Dict mapping task names to their CTAS node information:
            {
                'task_name': {
                    'node': 'node_name',
                    'uuid': 'node_uuid',
                    'hd4_pillars': ['pillar1', 'pillar2', ...]
                },
                ...
            }
        """
        # Dictionary stores subject mapping
        # Variable creates predicate structure
        # Tasks define object entries
        # Function returns subject result
        mapping: Dict[str, Dict[str, Any]] = {}
        try:
            # File provides subject data
            # Function reads predicate content
            # CSV contains object mappings
            # Dictionary stores subject results
            with open('task_map/task_rows.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Dictionary creates subject entry
                    # Mapping adds predicate information
                    # Task defines object key
                    # Structure organizes subject data
                    mapping[row['Task']] = {
                        'node': row['Node'],           # CTAS node name
                        'uuid': row['UUID'],           # Node UUID
                        'hd4_pillars': row['HD4_Pillars'].split(',')  # Associated pillars
                    }
        except Exception as e:
            # Logger reports subject error
            # Exception indicates predicate failure
            # Message contains object information
            # Function continues subject execution
            logging.error(f"Error loading task mapping: {e}")
        return mapping
    
    def _extract_codebase(self) -> bool:
        """
        Extract and filter codebase files
        
        # Function extracts subject codebase
        # Method filters predicate files
        # Repository contains object code
        # Analysis prepares subject data
        
        This method handles both zip files and directories:
        1. For zip files: Extracts only relevant files
        2. For directories: Copies only relevant files
        
        File filtering criteria:
        - Must have a code extension or no extension
        - Must not be in an excluded directory
        - Must be a file (not a directory)
        
        Returns:
            bool: True if extraction was successful, False otherwise
        """
        try:
            # Time measures subject performance
            # Variable tracks predicate execution
            # Function starts object timer
            # Analysis benchmarks subject operation
            start_time = time.time()
            print_progress(f"Checking input path: {self.input_path}")
            
            # Path represents subject location
            # Function checks predicate existence
            # File defines object input
            # Condition validates subject path
            if not self.input_path.exists():
                print_progress(f"Error: Input path does not exist: {self.input_path}")
                return False
            
            # Path identifies subject type
            # Extension determines predicate handling
            # Zip contains object files
            # Function processes subject accordingly
            if self.input_path.suffix == '.zip':
                # Function processes subject archive
                # Method extracts predicate files
                # Zip contains object code
                # Filter selects subject content
                print_progress("Input is a zip file, extracting...")
                with zipfile.ZipFile(self.input_path, 'r') as zip_ref:
                    # Function filters subject files
                    # List collects predicate entries
                    # Condition selects object items
                    # Comprehension builds subject collection
                    files_to_extract = [f for f in zip_ref.namelist() 
                                      if not any(excluded in f for excluded in self.exclude_dirs)
                                      and (Path(f).suffix in self.code_extensions or Path(f).suffix == '')]
                    total_files = len(files_to_extract)
                    print_progress(f"Found {total_files} relevant files in zip")
                    
                    # Loop processes subject files
                    # Function extracts predicate entries
                    # Archive contains object content
                    # Workspace receives subject files
                    for file in files_to_extract:
                        zip_ref.extract(file, self.workspace_dir)
                print_progress(f"Extracted codebase to {self.workspace_dir}")
            else:
                # Function processes subject directory
                # Method copies predicate files
                # Folder contains object code
                # Workspace receives subject content
                print_progress("Input is a directory, copying...")
                
                # Process collects subject files
                # Variables track predicate counts
                # List stores object references
                # Analysis prepares subject data
                relevant_files = []
                excluded_by_dir = 0
                excluded_by_ext = 0
                included_files = 0
                
                # Function scans subject files
                # Loop processes predicate entries
                # Directory contains object code
                # Filter selects subject content
                print_progress("Scanning files...")
                for src in self.input_path.rglob('*'):
                    if not src.is_file():
                        continue
                        
                    # Check identifies subject location
                    # Condition tests predicate directory
                    # Path contains object parts
                    # Filter excludes subject files
                    if any(excluded in src.parts for excluded in self.exclude_dirs):
                        excluded_by_dir += 1
                        if excluded_by_dir <= 5:  # Show first 5 exclusions
                            print_progress(f"Excluded by directory: {src}")
                        continue
                    
                    # Check examines subject extension
                    # Condition tests predicate type
                    # File contains object suffix
                    # Filter excludes subject files
                    if src.suffix not in self.code_extensions and src.suffix != '':
                        excluded_by_ext += 1
                        if excluded_by_ext <= 5:  # Show first 5 exclusions
                            print_progress(f"Excluded by extension: {src}")
                        continue
                    
                    # List includes subject file
                    # Collection adds predicate entry
                    # Path defines object location
                    # Filter accepts subject file
                    relevant_files.append(src)
                    included_files += 1
                    if included_files <= 5:  # Show first 5 inclusions
                        print_progress(f"Included: {src}")
                
                # Variable counts subject files
                # Output displays predicate summary
                # Count shows object totals
                # Function reports subject status
                total_files = len(relevant_files)
                print_progress(f"\nFile filtering summary:")
                print_progress(f"- Total files scanned: {excluded_by_dir + excluded_by_ext + included_files}")
                print_progress(f"- Excluded by directory: {excluded_by_dir}")
                print_progress(f"- Excluded by extension: {excluded_by_ext}")
                print_progress(f"- Included files: {included_files}")
                print_progress(f"Found {total_files} relevant files to copy")
                
                # Function cleans subject directory
                # Method removes predicate files
                # Workspace contains object content
                # System prepares subject location
                if self.workspace_dir.exists():
                    print_progress("Cleaning existing workspace directory...")
                    shutil.rmtree(self.workspace_dir)
                
                # Process copies subject files
                # Loop handles predicate entries
                # Path defines object location
                # Directory receives subject content
                copied = 0
                for src in relevant_files:
                    # Path creates subject destination
                    # Function builds predicate location
                    # File contains object content
                    # System preserves subject structure
                    dst = self.workspace_dir / src.relative_to(self.input_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1
                    if copied % 10 == 0:  # Show progress every 10 files
                        print_progress(f"Copied {copied}/{total_files} files...")
                
                print_progress(f"Copied {copied} files to {self.workspace_dir}")
            
            # Check verifies subject extraction
            # Condition tests predicate success
            # Directory contains object files
            # Function validates subject operation
            if not self.workspace_dir.exists():
                print_progress("Error: Workspace directory was not created")
                return False
            
            # Time measures subject performance
            # Variable tracks predicate completion
            # Function ends object timer
            # Output reports subject duration
            end_time = time.time()
            print_progress(f"Extraction completed in {end_time - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            print_progress(f"Error extracting codebase: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_structure(self) -> StructureInfo:
        """
        Analyze the structure of the extracted codebase
        
        # Function analyzes subject structure
        # Method examines predicate files
        # Codebase contains object patterns
        # Analysis produces subject results
        
        This method:
        1. Counts total files, code files, and artifact files
        2. Tracks file extensions
        3. Categorizes artifacts
        4. Finds most recent modification time
        
        Returns:
            StructureInfo containing:
            - files: Total number of files
            - code_files: Number of code files
            - artifact_files: Number of artifact files
            - extensions: Count of files by extension
            - artifacts: Artifact files by category
            - last_modified: Most recent file modification time
        """
        # Variables initialize subject counters
        # Function prepares predicate counts
        # Numbers track object quantities
        # Analysis uses subject counters
        file_count = 0          # Total files found
        code_file_count = 0     # Files with code extensions
        artifact_file_count = 0 # Files matching artifact categories
        extension_counter: Counter[str] = Counter()  # Count files by extension
        artifacts: Dict[str, List[str]] = {category: [] for category in self.artifact_categories.keys()}
        last_modified: Optional[float] = None  # Track most recent modification
        
        # Function analyzes subject files
        # Loop processes predicate paths
        # Workspace contains object content
        # Method examines subject structure
        for path in self.workspace_dir.rglob('*'):
            # Check filters subject paths
            # Condition tests predicate directory
            # Path contains object parts
            # Function skips subject exclusions
            if any(excluded in path.parts for excluded in self.exclude_dirs):
                continue
                
            # Check tests subject type
            # Condition verifies predicate file
            # Path represents object location
            # Function processes subject files
            if path.is_file():
                # Counter tracks subject files
                # Variable counts predicate extensions
                # Path contains object suffix
                # Analysis increments subject tallies
                file_count += 1
                ext = path.suffix.lower()
                extension_counter[ext] += 1
                
                # Check identifies subject type
                # Condition tests predicate extension
                # Set contains object values
                # Counter tracks subject code
                if ext in self.code_extensions:
                    code_file_count += 1
                
                # Loop categorizes subject files
                # Iteration processes predicate categories
                # Dictionary contains object extensions
                # Method classifies subject artifacts
                for category, extensions in self.artifact_categories.items():
                    if ext in extensions:
                        artifact_file_count += 1
                        artifacts[category].append(str(path.relative_to(self.workspace_dir)))
                        break
                
                # Stat retrieves subject metadata
                # Function gets predicate time
                # Path contains object file
                # Variable tracks subject modification
                mtime = path.stat().st_mtime
                if last_modified is None or mtime > last_modified:
                    last_modified = mtime
        
        # Dictionary returns subject results
        # Function assembles predicate data
        # Structure contains object information
        # Analysis produces subject output
        return {
            'files': file_count,
            'code_files': code_file_count,
            'artifact_files': artifact_file_count,
            'extensions': dict(extension_counter),
            'artifacts': artifacts,
            'last_modified': last_modified
        }
    
    def _extract_features(self) -> List[FeatureInfo]:
        """
        Extract features from code files by searching for keywords
        
        # Function extracts subject features
        # Method finds predicate patterns
        # Code contains object intelligence
        # Analysis discovers subject insights
        
        This method:
        1. Scans all code files in the workspace
        2. Searches for keywords in each category
        3. Captures context around each match
        4. Returns list of found features
        
        Returns:
            List[FeatureInfo] containing:
            - file: Path to file containing the feature
            - keyword: The matched keyword
            - category: Category of the keyword
            - context: Surrounding code context
        """
        # List stores subject features
        # Variable initializes predicate collection
        # Features contain object information
        # Function prepares subject results
        features: List[FeatureInfo] = []
        
        # Function processes subject files
        # Loop examines predicate code
        # Workspace contains object content
        # Method searches subject codebase
        for path in self.workspace_dir.rglob('*'):
            # Check filters subject files
            # Condition tests predicate type
            # Path contains object properties
            # Function skips subject non-code
            if not (path.is_file() and path.suffix in self.code_extensions):
                continue
                
            try:
                # Function reads subject content
                # Method opens predicate file
                # Path contains object location
                # Variable stores subject code
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Loop searches subject categories
                    # Iteration processes predicate keywords
                    # Dictionary contains object patterns
                    # Method finds subject features
                    for category, keywords in self.keyword_categories.items():
                        for keyword in keywords:
                            # Regex searches subject content
                            # Function finds predicate keyword
                            # Pattern defines object boundary
                            # Method locates subject match
                            if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                                # Dictionary records subject feature
                                # Function appends predicate match
                                # Feature contains object properties
                                # Collection grows subject results
                                features.append({
                                    'file': str(path.relative_to(self.workspace_dir)),
                                    'keyword': keyword,
                                    'category': category,
                                    'context': self._get_context(content, keyword)
                                })
            except Exception as e:
                # Logger reports subject error
                # Function handles predicate exception
                # Message contains object details
                # System continues subject processing
                logging.warning(f"Error processing {path}: {e}")
        
        # Function returns subject features
        # List contains predicate results
        # Collection holds object matches
        # Method delivers subject output
        return features
    
    def _get_context(self, content: str, keyword: str, context_lines: int = 3) -> str:
        """
        Get context around a keyword match in code
        
        # Function extracts subject context
        # Method finds predicate surroundings
        # Keyword marks object location
        # Lines provide subject information
        
        Args:
            content: The file content to search in
            keyword: The keyword to find context for
            context_lines: Number of lines to include before and after match
            
        Returns:
            String containing the matched line and surrounding context
        """
        # List splits subject content
        # Function divides predicate text
        # Lines contain object code
        # Variable stores subject segments
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Check finds subject keyword
            # Function tests predicate match
            # Line contains object text
            # Condition identifies subject location
            if keyword.lower() in line.lower():
                # Range determines subject boundaries
                # Function calculates predicate indexes
                # Context defines object scope
                # Slice selects subject lines
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return '\n'.join(lines[start:end])
        return ""
    
    def _map_to_ctas_nodes(self, features: List[FeatureInfo]) -> List[NodeMapping]:
        """
        Map extracted features to CTAS nodes
        
        # Function maps subject features
        # Method connects predicate patterns
        # Nodes represent object targets
        # Mapping creates subject relationships
        
        This method:
        1. Takes a list of found features
        2. Matches them against task mapping
        3. Creates node mappings for matches
        
        Args:
            features: List of features found in code
            
        Returns:
            List[NodeMapping] containing:
            - feature: The matched feature
            - node: CTAS node name
            - uuid: Node UUID
            - hd4_pillars: Associated HD4 pillars
        """
        # List stores subject mappings
        # Variable initializes predicate collection
        # Mappings contain object relationships
        # Function prepares subject results
        mapping: List[NodeMapping] = []
        for feature in features:
            # Loop matches subject features
            # Iteration processes predicate tasks
            # Dictionary contains object nodes
            # Method maps subject relationships
            for task, node_info in self.task_mapping.items():
                # Check tests subject match
                # Function compares predicate keyword
                # Task contains object name
                # Condition identifies subject relationship
                if feature['keyword'].lower() in task.lower():
                    # Dictionary creates subject mapping
                    # Function appends predicate match
                    # Mapping contains object properties
                    # Collection grows subject results
                    mapping.append({
                        'feature': feature,
                        'node': node_info['node'],
                        'uuid': node_info['uuid'],
                        'hd4_pillars': node_info['hd4_pillars']
                    })
        return mapping
    
    def _save_context(self, structure: StructureInfo, features: List[FeatureInfo], 
                     mapping: List[NodeMapping]) -> None:
        """
        Save analysis results and generate context summary
        
        # Function saves subject results
        # Method generates predicate files
        # Analysis contains object findings
        # Output preserves subject context
        
        This method:
        1. Saves detailed analysis results to JSON
        2. Generates a human-readable markdown summary
        3. Includes file counts, features, and node mappings
        
        Args:
            structure: Analysis of codebase structure
            features: List of found features
            mapping: CTAS node mappings
        """
        # Time creates subject timestamp
        # Function formats predicate string
        # Format represents object time
        # Variable stores subject identifier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # File receives subject analysis
        # Function saves predicate results
        # JSON contains object data
        # Output preserves subject findings
        analysis_file = self.context_dir / f'analysis_{timestamp}.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),  # Keep ISO format in JSON content
                'structure': structure,
                'features': features,
                'mapping': mapping
            }, f, indent=2)
        
        # File receives subject summary
        # Function generates predicate markdown
        # Content contains object results
        # Output presents subject findings
        summary_file = self.context_dir / f'context_{timestamp}.md'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Codebase Analysis Context
Timestamp: {datetime.now().isoformat()}

## Structure
- Total Files: {structure['files']}
- Code Files: {structure['code_files']}
- Artifact Files: {structure['artifact_files']}
- Top Extensions: {', '.join(f"{k}: {v}" for k, v in sorted(structure['extensions'].items(), key=lambda x: x[1], reverse=True)[:5])}

## Artifacts
{"".join(f"### {category}\n" + "".join(f"- {file}\n" for file in files[:5]) for category, files in structure['artifacts'].items() if files)}

## Key Features
{"".join(f"- {f['keyword']} ({f['category']}) in {f['file']}\n" for f in features[:10])}

## CTAS Node Mapping
{"".join(f"- {m['node']} (UUID: {m['uuid']})\n" for m in mapping[:5])}

## Development Context
This analysis was performed on the codebase at {self.input_path}.
The workspace is available at {self.workspace_dir}.
Full analysis results are stored in {analysis_file}.

Use this context to maintain development continuity and prevent context loss.
""")
    
    def analyze(self) -> bool:
        """
        Run the complete analysis workflow
        
        # Function executes subject workflow
        # Method coordinates predicate steps
        # Analysis performs object operations
        # Process produces subject results
        
        This method:
        1. Extracts and filters the codebase
        2. Analyzes codebase structure
        3. Extracts features from code
        4. Maps features to CTAS nodes
        5. Saves analysis results
        
        Returns:
            bool: True if analysis completed successfully, False otherwise
        """
        try:
            # Variables configure subject workflow
            # Count defines predicate steps
            # Numbers track object progress
            # Function prepares subject execution
            total_steps = 5
            current_step = 1
            
            print_progress("Starting analysis...")
            
            # Step extracts subject codebase
            # Function performs predicate operation
            # Number tracks object progress
            # Method processes subject files
            print_step(current_step, total_steps, "Extracting codebase...")
            if not self._extract_codebase():
                print_progress("Failed to extract codebase")
                return False
            current_step += 1
            
            # Step analyzes subject structure
            # Function performs predicate operation
            # Number tracks object progress
            # Method examines subject files
            print_step(current_step, total_steps, "Analyzing structure...")
            structure = self._analyze_structure()
            print_progress(f"Found {structure['files']} files, {structure['code_files']} code files")
            current_step += 1
            
            # Step extracts subject features
            # Function performs predicate operation
            # Number tracks object progress
            # Method identifies subject patterns
            print_step(current_step, total_steps, "Extracting features...")
            features = self._extract_features()
            print_progress(f"Found {len(features)} features")
            current_step += 1
            
            # Step maps subject features
            # Function performs predicate operation
            # Number tracks object progress
            # Method connects subject nodes
            print_step(current_step, total_steps, "Mapping to CTAS nodes...")
            mapping = self._map_to_ctas_nodes(features)
            print_progress(f"Mapped {len(mapping)} features to CTAS nodes")
            current_step += 1
            
            # Step saves subject context
            # Function performs predicate operation
            # Number tracks object progress
            # Method preserves subject results
            print_step(current_step, total_steps, "Saving context...")
            self._save_context(structure, features, mapping)
            print_progress("Context saved")
            
            return True
            
        except Exception as e:
            print_progress(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Function defines subject entry
# Method provides predicate interface
# Arguments supply object parameters
# Script executes subject analysis
def main() -> None:
    """
    Main entry point for the analysis script
    
    # Function provides subject entry
    # Method creates predicate interface
    # Script defines object workflow
    # Main executes subject analysis
    
    This function:
    1. Parses command line arguments
    2. Creates analyzer instance
    3. Runs analysis
    4. Handles errors and exit codes
    """
    # Module imports subject components
    # Script loads predicate libraries
    # Packages supply object functionality
    # Function prepares subject execution
    import argparse
    import sys
    import traceback
    
    try:
        print_progress("Initializing...")
        
        # Parser processes subject arguments
        # Function creates predicate interface
        # Arguments supply object parameters
        # Script configures subject execution
        parser = argparse.ArgumentParser(description='Analyze codebase and preserve context')
        parser.add_argument('--input', required=True, help='Path to input codebase (zip or directory)')
        parser.add_argument('--output', required=True, help='Output directory')
        args = parser.parse_args()
        
        print_progress(f"Starting analysis with:")
        print_progress(f"  Input path: {args.input}")
        print_progress(f"  Output dir: {args.output}")
        
        # Function creates subject analyzer
        # Constructor initializes predicate instance
        # Parameters supply object configuration
        # Variable stores subject reference
        analyzer = CodebaseAnalyzer(args.input, args.output)
        if analyzer.analyze():
            print_progress("Analysis completed successfully")
        else:
            print_progress("Analysis failed")
            sys.exit(1)
            
    except Exception as e:
        print_progress(f"Error: {str(e)}")
        print_progress("Traceback:")
        traceback.print_exc()
        sys.exit(1)

# Condition checks subject execution
# Script verifies predicate environment
# Name identifies object context
# Module runs subject function
if __name__ == '__main__':
    main()