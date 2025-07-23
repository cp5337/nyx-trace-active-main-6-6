#!/usr/bin/env python3
"""
Team B Large Files Management Implementation
==========================================

This module handles large file detection, management, optimization, and archiving
for the nyx-trace repository.

Key Features:
- Large file detection and analysis
- File size optimization and compression
- Automated archiving and cleanup
- Performance monitoring and reporting
"""

import os
import sys
import gzip
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LargeFilesManager:
    """
    Comprehensive large files management system
    """
    
    def __init__(self, repo_root=None, size_threshold_mb=1):
        self.repo_root = Path(repo_root) if repo_root else Path('.').resolve()
        self.size_threshold = size_threshold_mb * 1024 * 1024  # Convert MB to bytes
        self.excluded_patterns = {
            '.git', '__pycache__', '.pyc', '.venv', 'node_modules',
            '.DS_Store', '.pytest_cache', '.coverage'
        }
        self.archive_dir = self.repo_root / 'archived_files'
        self.reports_dir = self.repo_root / 'file_reports'
        
        # Create directories if they don't exist
        self.archive_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Large Files Manager for: {self.repo_root}")
        logger.info(f"Size threshold: {size_threshold_mb}MB ({self.size_threshold} bytes)")
    
    def scan_large_files(self) -> List[Dict]:
        """
        Scan repository for files exceeding size threshold
        
        Returns:
            List of dictionaries containing file metadata
        """
        logger.info("Scanning for large files...")
        large_files = []
        
        for file_path in self._get_all_files():
            try:
                file_size = file_path.stat().st_size
                
                if file_size > self.size_threshold:
                    file_info = {
                        'path': str(file_path.relative_to(self.repo_root)),
                        'absolute_path': str(file_path),
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'extension': file_path.suffix,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        'hash': self._calculate_file_hash(file_path)
                    }
                    large_files.append(file_info)
                    
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not access {file_path}: {e}")
        
        logger.info(f"Found {len(large_files)} large files")
        return large_files
    
    def analyze_file_patterns(self, large_files: List[Dict]) -> Dict:
        """
        Analyze patterns in large files
        
        Args:
            large_files: List of large file metadata
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing file patterns...")
        
        # Group by extension
        by_extension = {}
        by_directory = {}
        total_size = 0
        
        for file_info in large_files:
            ext = file_info['extension'] or 'no_extension'
            size = file_info['size_bytes']
            dir_path = str(Path(file_info['path']).parent)
            
            # By extension
            if ext not in by_extension:
                by_extension[ext] = {'count': 0, 'total_size': 0, 'files': []}
            by_extension[ext]['count'] += 1
            by_extension[ext]['total_size'] += size
            by_extension[ext]['files'].append(file_info['path'])
            
            # By directory
            if dir_path not in by_directory:
                by_directory[dir_path] = {'count': 0, 'total_size': 0, 'files': []}
            by_directory[dir_path]['count'] += 1
            by_directory[dir_path]['total_size'] += size
            by_directory[dir_path]['files'].append(file_info['path'])
            
            total_size += size
        
        # Sort by size
        by_extension = dict(sorted(by_extension.items(), 
                                 key=lambda x: x[1]['total_size'], reverse=True))
        by_directory = dict(sorted(by_directory.items(), 
                                 key=lambda x: x[1]['total_size'], reverse=True))
        
        analysis = {
            'total_files': len(large_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_extension': by_extension,
            'by_directory': by_directory,
            'largest_files': sorted(large_files, key=lambda x: x['size_bytes'], reverse=True)[:10]
        }
        
        return analysis
    
    def optimize_files(self, large_files: List[Dict], dry_run=True) -> Dict:
        """
        Optimize large files through compression and other techniques
        
        Args:
            large_files: List of large file metadata
            dry_run: If True, only simulate optimization
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Optimizing files (dry_run={dry_run})...")
        
        optimization_results = {
            'processed': 0,
            'compressed': 0,
            'archived': 0,
            'deleted': 0,
            'total_saved_bytes': 0,
            'actions': []
        }
        
        for file_info in large_files:
            file_path = Path(file_info['absolute_path'])
            
            if not file_path.exists():
                continue
                
            action = self._determine_optimization_action(file_info)
            optimization_results['actions'].append({
                'file': file_info['path'],
                'action': action,
                'original_size': file_info['size_bytes']
            })
            
            if not dry_run:
                if action == 'compress':
                    saved_bytes = self._compress_file(file_path)
                    optimization_results['compressed'] += 1
                    optimization_results['total_saved_bytes'] += saved_bytes
                    
                elif action == 'archive':
                    self._archive_file(file_path)
                    optimization_results['archived'] += 1
                    
                elif action == 'delete':
                    file_path.unlink()
                    optimization_results['deleted'] += 1
                    optimization_results['total_saved_bytes'] += file_info['size_bytes']
            
            optimization_results['processed'] += 1
        
        return optimization_results
    
    def generate_report(self, large_files: List[Dict], analysis: Dict, optimization: Dict) -> str:
        """
        Generate comprehensive large files report
        
        Args:
            large_files: List of large file metadata
            analysis: File pattern analysis results
            optimization: Optimization results
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating large files report...")
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'repository': str(self.repo_root),
            'size_threshold_mb': self.size_threshold / (1024 * 1024),
            'large_files': large_files,
            'analysis': analysis,
            'optimization': optimization
        }
        
        # Generate JSON report
        json_report_path = self.reports_dir / f"large_files_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate markdown report
        md_report_path = self.reports_dir / f"large_files_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        markdown_content = self._generate_markdown_report(report_data)
        with open(md_report_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Reports generated: {json_report_path}, {md_report_path}")
        return str(md_report_path)
    
    def _get_all_files(self):
        """Get all files in repository, excluding patterns"""
        for root, dirs, files in os.walk(self.repo_root):
            # Remove excluded directories from dirs list
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.excluded_patterns)]
            
            for file in files:
                file_path = Path(root) / file
                # Skip excluded file patterns
                if not any(pattern in str(file_path) for pattern in self.excluded_patterns):
                    yield file_path
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]  # Truncated for readability
        except Exception:
            return "unknown"
    
    def _determine_optimization_action(self, file_info: Dict) -> str:
        """Determine appropriate optimization action for file"""
        ext = file_info['extension'].lower()
        size_mb = file_info['size_mb']
        
        # Python cache files - delete
        if ext in ['.pyc', '.pyo'] or '__pycache__' in file_info['path']:
            return 'delete'
        
        # Log files - compress or archive
        if ext in ['.log', '.txt'] and size_mb > 5:
            return 'compress'
        
        # Data files - archive if very large
        if ext in ['.csv', '.json', '.xml'] and size_mb > 10:
            return 'archive'
        
        # Media files - archive
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.mp4', '.avi'] and size_mb > 5:
            return 'archive'
        
        # Keep everything else
        return 'keep'
    
    def _compress_file(self, file_path: Path) -> int:
        """Compress file using gzip"""
        original_size = file_path.stat().st_size
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original file
        file_path.unlink()
        
        compressed_size = compressed_path.stat().st_size
        saved_bytes = original_size - compressed_size
        
        logger.info(f"Compressed {file_path.name}: {original_size} -> {compressed_size} bytes (saved {saved_bytes})")
        return saved_bytes
    
    def _archive_file(self, file_path: Path):
        """Move file to archive directory"""
        archive_path = self.archive_dir / file_path.name
        
        # Ensure unique filename
        counter = 1
        while archive_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            archive_path = self.archive_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(file_path), str(archive_path))
        logger.info(f"Archived {file_path.name} to {archive_path}")
    
    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Generate markdown formatted report"""
        analysis = report_data['analysis']
        optimization = report_data['optimization']
        
        markdown = f"""# Large Files Report
        
Generated: {report_data['timestamp']}
Repository: {report_data['repository']}
Size Threshold: {report_data['size_threshold_mb']}MB

## Summary

- **Total Large Files:** {analysis['total_files']}
- **Total Size:** {analysis['total_size_mb']}MB ({analysis['total_size_bytes']:,} bytes)
- **Files Processed:** {optimization['processed']}
- **Space Saved:** {optimization['total_saved_bytes'] / (1024*1024):.2f}MB

## Largest Files

| File | Size (MB) | Extension | Modified |
|------|-----------|-----------|----------|
"""
        
        for file_info in analysis['largest_files'][:10]:
            markdown += f"| {file_info['path']} | {file_info['size_mb']} | {file_info['extension']} | {file_info['modified'][:10]} |\n"
        
        markdown += "\n## Files by Extension\n\n"
        for ext, data in list(analysis['by_extension'].items())[:10]:
            markdown += f"- **{ext}**: {data['count']} files, {data['total_size'] / (1024*1024):.2f}MB\n"
        
        markdown += "\n## Files by Directory\n\n"
        for dir_path, data in list(analysis['by_directory'].items())[:10]:
            markdown += f"- **{dir_path}**: {data['count']} files, {data['total_size'] / (1024*1024):.2f}MB\n"
        
        if optimization['actions']:
            markdown += "\n## Optimization Actions\n\n"
            for action in optimization['actions'][:20]:
                markdown += f"- **{action['action']}**: {action['file']} ({action['original_size'] / (1024*1024):.2f}MB)\n"
        
        return markdown

    def run_full_analysis(self, dry_run=True):
        """Run complete large files analysis and optimization"""
        logger.info("🚀 Starting Team B Large Files Analysis...")
        
        # Scan for large files
        large_files = self.scan_large_files()
        
        if not large_files:
            logger.info("✅ No large files found above threshold")
            return
        
        # Analyze patterns
        analysis = self.analyze_file_patterns(large_files)
        
        # Optimize files
        optimization = self.optimize_files(large_files, dry_run=dry_run)
        
        # Generate report
        report_path = self.generate_report(large_files, analysis, optimization)
        
        logger.info(f"📊 Analysis complete. Report saved to: {report_path}")
        
        # Summary
        logger.info(f"🎯 Team B Large Files Summary:")
        logger.info(f"   - Large files found: {len(large_files)}")
        logger.info(f"   - Total size: {analysis['total_size_mb']}MB")
        logger.info(f"   - Files processed: {optimization['processed']}")
        if not dry_run:
            logger.info(f"   - Space saved: {optimization['total_saved_bytes'] / (1024*1024):.2f}MB")
        
        return {
            'large_files': large_files,
            'analysis': analysis,
            'optimization': optimization,
            'report_path': report_path
        }

if __name__ == "__main__":
    manager = LargeFilesManager()
    results = manager.run_full_analysis(dry_run=True)
    
    if results:
        print("🎉 Team B Large Files Management completed successfully!")
    else:
        print("✅ No large files management needed.")
