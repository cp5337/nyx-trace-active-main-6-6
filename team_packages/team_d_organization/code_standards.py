#!/usr/bin/env python3
"""
Team D Code Standards Checker
============================

This module provides comprehensive code standards checking for the nyx-trace repository.
It enforces the CTAS standards including line length, comment density, and module size.

Key Features:
- Line length compliance (80 characters)
- Comment density analysis (15% minimum)
- Module size limits (300 lines max)
- Docstring coverage analysis
- Code quality metrics
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeStandardsChecker:
    """
    Comprehensive code standards checker for CTAS compliance
    """
    
    def __init__(self, repo_root=None):
        self.repo_root = Path(repo_root) if repo_root else Path('.').resolve()
        self.excluded_patterns = {
            '.git', '__pycache__', '.pyc', '.venv', 'node_modules',
            '.DS_Store', '.pytest_cache', '.coverage', 'archived_files'
        }
        
        # CTAS Standards
        self.max_line_length = 80
        self.min_comment_density = 15.0  # percentage
        self.max_module_size = 300  # lines
        self.require_docstrings = True
        
        logger.info(f"Initialized Code Standards Checker for: {self.repo_root}")
        logger.info(f"Standards: {self.max_line_length} chars, {self.min_comment_density}% comments, {self.max_module_size} lines max")
    
    def check_repository(self) -> Dict:
        """
        Check entire repository for code standards compliance
        
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info("🔍 Checking repository code standards...")
        
        python_files = self._get_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        results = {
            'summary': {
                'total_files': len(python_files),
                'compliant_files': 0,
                'non_compliant_files': 0,
                'compliance_rate': 0.0
            },
            'standards': {
                'line_length': {'compliant': 0, 'non_compliant': 0},
                'comment_density': {'compliant': 0, 'non_compliant': 0},
                'module_size': {'compliant': 0, 'non_compliant': 0},
                'docstrings': {'compliant': 0, 'non_compliant': 0}
            },
            'file_details': [],
            'recommendations': []
        }
        
        for file_path in python_files:
            file_result = self._check_file(file_path)
            results['file_details'].append(file_result)
            
            # Update summary statistics
            if file_result['overall_compliant']:
                results['summary']['compliant_files'] += 1
            else:
                results['summary']['non_compliant_files'] += 1
            
            # Update standards statistics
            for standard in ['line_length', 'comment_density', 'module_size', 'docstrings']:
                if file_result['standards'][standard]['compliant']:
                    results['standards'][standard]['compliant'] += 1
                else:
                    results['standards'][standard]['non_compliant'] += 1
        
        # Calculate compliance rates
        total_files = len(python_files)
        if total_files > 0:
            results['summary']['compliance_rate'] = (
                results['summary']['compliant_files'] / total_files * 100
            )
            
            for standard in results['standards']:
                total = results['standards'][standard]['compliant'] + results['standards'][standard]['non_compliant']
                if total > 0:
                    compliance_rate = results['standards'][standard]['compliant'] / total * 100
                    results['standards'][standard]['compliance_rate'] = compliance_rate
                else:
                    results['standards'][standard]['compliance_rate'] = 100.0
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        logger.info(f"✅ Code standards analysis complete")
        logger.info(f"   Overall compliance: {results['summary']['compliance_rate']:.1f}%")
        logger.info(f"   Compliant files: {results['summary']['compliant_files']}/{total_files}")
        
        return results
    
    def _check_file(self, file_path: Path) -> Dict:
        """
        Check a single Python file for standards compliance
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary containing file analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            file_result = {
                'path': str(file_path.relative_to(self.repo_root)),
                'absolute_path': str(file_path),
                'total_lines': len(lines),
                'standards': {},
                'overall_compliant': True,
                'issues': []
            }
            
            # Check line length
            line_length_result = self._check_line_length(lines)
            file_result['standards']['line_length'] = line_length_result
            if not line_length_result['compliant']:
                file_result['overall_compliant'] = False
                file_result['issues'].extend(line_length_result['issues'])
            
            # Check comment density
            comment_density_result = self._check_comment_density(lines)
            file_result['standards']['comment_density'] = comment_density_result
            if not comment_density_result['compliant']:
                file_result['overall_compliant'] = False
                file_result['issues'].extend(comment_density_result['issues'])
            
            # Check module size
            module_size_result = self._check_module_size(lines)
            file_result['standards']['module_size'] = module_size_result
            if not module_size_result['compliant']:
                file_result['overall_compliant'] = False
                file_result['issues'].extend(module_size_result['issues'])
            
            # Check docstrings
            docstring_result = self._check_docstrings(file_path, lines)
            file_result['standards']['docstrings'] = docstring_result
            if not docstring_result['compliant']:
                file_result['overall_compliant'] = False
                file_result['issues'].extend(docstring_result['issues'])
            
            return file_result
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return {
                'path': str(file_path.relative_to(self.repo_root)),
                'error': str(e),
                'overall_compliant': False,
                'standards': {
                    'line_length': {'compliant': False},
                    'comment_density': {'compliant': False},
                    'module_size': {'compliant': False},
                    'docstrings': {'compliant': False}
                }
            }
    
    def _check_line_length(self, lines: List[str]) -> Dict:
        """Check line length compliance"""
        long_lines = []
        
        for i, line in enumerate(lines, 1):
            if len(line.rstrip('\n')) > self.max_line_length:
                long_lines.append({
                    'line_number': i,
                    'length': len(line.rstrip('\n')),
                    'content': line[:50] + '...' if len(line) > 50 else line.rstrip('\n')
                })
        
        compliant = len(long_lines) == 0
        issues = []
        
        if not compliant:
            issues.append(f"Found {len(long_lines)} lines exceeding {self.max_line_length} characters")
            for long_line in long_lines[:5]:  # Show first 5 issues
                issues.append(f"  Line {long_line['line_number']}: {long_line['length']} chars")
        
        return {
            'compliant': compliant,
            'long_lines': long_lines,
            'issues': issues
        }
    
    def _check_comment_density(self, lines: List[str]) -> Dict:
        """Check comment density compliance"""
        comment_lines = 0
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                comment_lines += 1
            elif stripped:  # Non-empty line that's not a comment
                code_lines += 1
        
        total_lines = comment_lines + code_lines
        comment_density = (comment_lines / total_lines * 100) if total_lines > 0 else 0
        
        compliant = comment_density >= self.min_comment_density
        issues = []
        
        if not compliant:
            issues.append(f"Comment density {comment_density:.1f}% below minimum {self.min_comment_density}%")
            issues.append(f"  Comment lines: {comment_lines}, Code lines: {code_lines}")
        
        return {
            'compliant': compliant,
            'comment_density': comment_density,
            'comment_lines': comment_lines,
            'code_lines': code_lines,
            'issues': issues
        }
    
    def _check_module_size(self, lines: List[str]) -> Dict:
        """Check module size compliance"""
        module_size = len(lines)
        compliant = module_size <= self.max_module_size
        issues = []
        
        if not compliant:
            issues.append(f"Module size {module_size} lines exceeds maximum {self.max_module_size} lines")
        
        return {
            'compliant': compliant,
            'module_size': module_size,
            'issues': issues
        }
    
    def _check_docstrings(self, file_path: Path, lines: List[str]) -> Dict:
        """Check docstring coverage"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            functions_with_docstrings = sum(1 for f in functions if ast.get_docstring(f))
            classes_with_docstrings = sum(1 for c in classes if ast.get_docstring(c))
            
            total_elements = len(functions) + len(classes)
            elements_with_docstrings = functions_with_docstrings + classes_with_docstrings
            
            docstring_coverage = (elements_with_docstrings / total_elements * 100) if total_elements > 0 else 100
            
            # Require at least 80% docstring coverage
            compliant = docstring_coverage >= 80.0
            issues = []
            
            if not compliant:
                issues.append(f"Docstring coverage {docstring_coverage:.1f}% below minimum 80%")
                issues.append(f"  Functions: {functions_with_docstrings}/{len(functions)} with docstrings")
                issues.append(f"  Classes: {classes_with_docstrings}/{len(classes)} with docstrings")
            
            return {
                'compliant': compliant,
                'docstring_coverage': docstring_coverage,
                'functions': len(functions),
                'functions_with_docstrings': functions_with_docstrings,
                'classes': len(classes),
                'classes_with_docstrings': classes_with_docstrings,
                'issues': issues
            }
            
        except Exception as e:
            logger.warning(f"Could not parse {file_path} for docstring analysis: {e}")
            return {
                'compliant': False,
                'error': str(e),
                'issues': [f"Could not analyze docstrings: {e}"]
            }
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in repository"""
        python_files = []
        
        for file_path in self.repo_root.rglob('*.py'):
            # Skip excluded patterns
            if any(pattern in str(file_path) for pattern in self.excluded_patterns):
                continue
            
            python_files.append(file_path)
        
        return sorted(python_files)
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Overall recommendations
        if results['summary']['compliance_rate'] < 80:
            recommendations.append("Overall compliance below 80% - prioritize standards enforcement")
        
        # Line length recommendations
        line_length_non_compliant = results['standards']['line_length']['non_compliant']
        if line_length_non_compliant > 0:
            recommendations.append(f"Fix {line_length_non_compliant} files with long lines (>80 chars)")
        
        # Comment density recommendations
        comment_density_non_compliant = results['standards']['comment_density']['non_compliant']
        if comment_density_non_compliant > 0:
            recommendations.append(f"Add comments to {comment_density_non_compliant} files (<15% comment density)")
        
        # Module size recommendations
        module_size_non_compliant = results['standards']['module_size']['non_compliant']
        if module_size_non_compliant > 0:
            recommendations.append(f"Split {module_size_non_compliant} large files (>300 lines)")
        
        # Docstring recommendations
        docstring_non_compliant = results['standards']['docstrings']['non_compliant']
        if docstring_non_compliant > 0:
            recommendations.append(f"Add docstrings to {docstring_non_compliant} files (<80% coverage)")
        
        # Specific file recommendations
        large_files = [f for f in results['file_details'] 
                      if not f.get('standards', {}).get('module_size', {}).get('compliant', True)]
        
        if large_files:
            recommendations.append("Large files to split:")
            for file_info in large_files[:5]:  # Show top 5
                size = file_info.get('standards', {}).get('module_size', {}).get('module_size', 0)
                recommendations.append(f"  - {file_info['path']}: {size} lines")
        
        return recommendations
    
    def generate_report(self, results: Dict) -> str:
        """Generate markdown report from analysis results"""
        report_lines = [
            "# Code Standards Analysis Report",
            "",
            f"**Repository**: {self.repo_root}",
            f"**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Files**: {results['summary']['total_files']}",
            f"- **Compliant Files**: {results['summary']['compliant_files']}",
            f"- **Non-Compliant Files**: {results['summary']['non_compliant_files']}",
            f"- **Overall Compliance**: {results['summary']['compliance_rate']:.1f}%",
            "",
            "## Standards Compliance",
            "",
            "| Standard | Compliant | Non-Compliant | Compliance Rate |",
            "|----------|-----------|---------------|-----------------|"
        ]
        
        for standard, data in results['standards'].items():
            compliant = data['compliant']
            non_compliant = data['non_compliant']
            total = compliant + non_compliant
            rate = (compliant / total * 100) if total > 0 else 100.0
            report_lines.append(f"| {standard.replace('_', ' ').title()} | {compliant} | {non_compliant} | {rate:.1f}% |")
        
        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        for recommendation in results['recommendations']:
            report_lines.append(f"- {recommendation}")
        
        return "\n".join(report_lines)
