#!/usr/bin/env python3
"""
CTAS Enhanced Code Standards Enforcement System
==============================================

A comprehensive, well-organized code standards enforcement system for the CTAS project.
Provides detailed analysis, actionable recommendations, and automated compliance tracking.

Key Features:
- Multi-standard compliance checking
- Detailed reporting with actionable insights
- Automated fix suggestions
- Progress tracking and metrics
- Integration with CTAS team workflows
"""

import os
import re
import ast
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandardType(Enum):
    """Types of code standards"""
    LINE_LENGTH = "line_length"
    COMMENT_DENSITY = "comment_density"
    MODULE_SIZE = "module_size"
    DOCSTRINGS = "docstrings"
    NAMING_CONVENTIONS = "naming_conventions"
    COMPLEXITY = "complexity"

class ComplianceLevel(Enum):
    """Compliance levels"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # 30-49%
    CRITICAL = "critical"   # 0-29%

@dataclass
class StandardConfig:
    """Configuration for a code standard"""
    name: str
    description: str
    enabled: bool = True
    weight: float = 1.0
    threshold: float = 80.0
    max_value: Optional[int] = None
    min_value: Optional[float] = None

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: str
    absolute_path: str
    total_lines: int
    standards: Dict[str, Dict[str, Any]]
    overall_compliant: bool
    compliance_score: float
    issues: List[str]
    recommendations: List[str]
    priority: str  # high, medium, low

@dataclass
class RepositoryAnalysis:
    """Complete repository analysis results"""
    summary: Dict[str, Any]
    standards: Dict[str, Dict[str, Any]]
    file_details: List[FileAnalysis]
    recommendations: List[str]
    progress_metrics: Dict[str, Any]
    compliance_level: ComplianceLevel

class CTASCodeStandards:
    """
    Enhanced CTAS Code Standards Enforcement System
    
    Provides comprehensive code quality analysis with actionable insights
    and automated recommendations for the CTAS project.
    """
    
    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = Path(repo_root) if repo_root else Path('.').resolve()
        
        # Exclusion patterns
        self.excluded_patterns = {
            '.git', '__pycache__', '.pyc', '.venv', 'node_modules',
            '.DS_Store', '.pytest_cache', '.coverage', 'archived_files',
            'handoff_package', 'file_reports', 'test_*.py'
        }
        
        # CTAS Standards Configuration
        self.standards_config = {
            StandardType.LINE_LENGTH: StandardConfig(
                name="Line Length",
                description="Maximum line length of 80 characters",
                enabled=True,
                weight=1.0,
                threshold=80.0,
                max_value=80
            ),
            StandardType.COMMENT_DENSITY: StandardConfig(
                name="Comment Density",
                description="Minimum 15% comment density",
                enabled=True,
                weight=1.0,
                threshold=80.0,
                min_value=15.0
            ),
            StandardType.MODULE_SIZE: StandardConfig(
                name="Module Size",
                description="Maximum 300 lines per module",
                enabled=True,
                weight=1.0,
                threshold=80.0,
                max_value=300
            ),
            StandardType.DOCSTRINGS: StandardConfig(
                name="Docstring Coverage",
                description="Minimum 80% docstring coverage",
                enabled=True,
                weight=1.0,
                threshold=80.0,
                min_value=80.0
            ),
            StandardType.NAMING_CONVENTIONS: StandardConfig(
                name="Naming Conventions",
                description="Python naming convention compliance",
                enabled=True,
                weight=0.8,
                threshold=80.0
            ),
            StandardType.COMPLEXITY: StandardConfig(
                name="Code Complexity",
                description="Cyclomatic complexity analysis",
                enabled=True,
                weight=0.9,
                threshold=80.0
            )
        }
        
        logger.info(f"🚀 Initialized CTAS Code Standards for: {self.repo_root}")
        logger.info(f"📊 Standards: {len(self.standards_config)} active standards")
    
    def analyze_repository(self) -> RepositoryAnalysis:
        """
        Perform comprehensive repository analysis
        
        Returns:
            RepositoryAnalysis with complete results
        """
        logger.info("🔍 Starting comprehensive CTAS code standards analysis...")
        
        start_time = time.time()
        python_files = self._get_python_files()
        logger.info(f"📁 Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        file_analyses = []
        standards_summary = defaultdict(lambda: {'compliant': 0, 'non_compliant': 0, 'total': 0})
        
        for file_path in python_files:
            file_analysis = self._analyze_file(file_path)
            file_analyses.append(file_analysis)
            
            # Update standards summary
            for standard_type, result in file_analysis.standards.items():
                standards_summary[standard_type]['total'] += 1
                if result.get('compliant', False):
                    standards_summary[standard_type]['compliant'] += 1
                else:
                    standards_summary[standard_type]['non_compliant'] += 1
        
        # Calculate overall metrics
        total_files = len(python_files)
        compliant_files = sum(1 for f in file_analyses if f.overall_compliant)
        overall_compliance = (compliant_files / total_files * 100) if total_files > 0 else 0
        
        # Determine compliance level
        compliance_level = self._get_compliance_level(overall_compliance)
        
        # Calculate standards compliance rates
        standards_compliance = {}
        for standard_type, summary in standards_summary.items():
            total = summary['total']
            compliant = summary['compliant']
            compliance_rate = (compliant / total * 100) if total > 0 else 100.0
            standards_compliance[standard_type] = {
                'compliant': compliant,
                'non_compliant': summary['non_compliant'],
                'compliance_rate': compliance_rate,
                'total': total
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_analyses, standards_compliance)
        
        # Calculate progress metrics
        progress_metrics = self._calculate_progress_metrics(file_analyses, standards_compliance)
        
        analysis_time = time.time() - start_time
        
        logger.info(f"✅ Analysis complete in {analysis_time:.2f}s")
        logger.info(f"📊 Overall compliance: {overall_compliance:.1f}% ({compliance_level.value})")
        logger.info(f"📈 Compliant files: {compliant_files}/{total_files}")
        
        return RepositoryAnalysis(
            summary={
                'total_files': total_files,
                'compliant_files': compliant_files,
                'non_compliant_files': total_files - compliant_files,
                'compliance_rate': overall_compliance,
                'analysis_time': analysis_time,
                'compliance_level': compliance_level.value
            },
            standards=standards_compliance,
            file_details=file_analyses,
            recommendations=recommendations,
            progress_metrics=progress_metrics,
            compliance_level=compliance_level
        )
    
    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            standards_results = {}
            issues = []
            recommendations = []
            
            # Check each standard
            for standard_type, config in self.standards_config.items():
                if not config.enabled:
                    continue
                
                result = self._check_standard(standard_type, file_path, lines)
                standards_results[standard_type.value] = result
                
                if not result.get('compliant', True):
                    issues.extend(result.get('issues', []))
                    recommendations.extend(result.get('recommendations', []))
            
            # Calculate overall compliance score
            compliance_score = self._calculate_compliance_score(standards_results)
            overall_compliant = compliance_score >= 80.0
            
            # Determine priority
            priority = self._determine_priority(standards_results, compliance_score)
            
            return FileAnalysis(
                path=str(file_path.relative_to(self.repo_root)),
                absolute_path=str(file_path),
                total_lines=len(lines),
                standards=standards_results,
                overall_compliant=overall_compliant,
                compliance_score=compliance_score,
                issues=issues,
                recommendations=recommendations,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return FileAnalysis(
                path=str(file_path.relative_to(self.repo_root)),
                absolute_path=str(file_path),
                total_lines=0,
                standards={},
                overall_compliant=False,
                compliance_score=0.0,
                issues=[f"Analysis error: {e}"],
                recommendations=["Fix file parsing issues"],
                priority="high"
            )
    
    def _check_standard(self, standard_type: StandardType, file_path: Path, lines: List[str]) -> Dict[str, Any]:
        """Check a specific standard"""
        if standard_type == StandardType.LINE_LENGTH:
            return self._check_line_length(lines)
        elif standard_type == StandardType.COMMENT_DENSITY:
            return self._check_comment_density(lines)
        elif standard_type == StandardType.MODULE_SIZE:
            return self._check_module_size(lines)
        elif standard_type == StandardType.DOCSTRINGS:
            return self._check_docstrings(file_path, lines)
        elif standard_type == StandardType.NAMING_CONVENTIONS:
            return self._check_naming_conventions(file_path, lines)
        elif standard_type == StandardType.COMPLEXITY:
            return self._check_complexity(file_path, lines)
        else:
            return {'compliant': True, 'issues': [], 'recommendations': []}
    
    def _check_line_length(self, lines: List[str]) -> Dict[str, Any]:
        """Check line length compliance"""
        max_length = self.standards_config[StandardType.LINE_LENGTH].max_value
        long_lines = []
        
        for i, line in enumerate(lines, 1):
            line_length = len(line.rstrip('\n'))
            if line_length > max_length:
                long_lines.append({
                    'line_number': i,
                    'length': line_length,
                    'content': line[:50] + '...' if line_length > 50 else line.rstrip('\n')
                })
        
        compliant = len(long_lines) == 0
        issues = []
        recommendations = []
        
        if not compliant:
            issues.append(f"Found {len(long_lines)} lines exceeding {max_length} characters")
            recommendations.append(f"Break long lines into multiple lines or use line continuation")
            
            # Show examples
            for long_line in long_lines[:3]:
                issues.append(f"  Line {long_line['line_number']}: {long_line['length']} chars")
        
        return {
            'compliant': compliant,
            'long_lines': long_lines,
            'issues': issues,
            'recommendations': recommendations,
            'metric': len(long_lines)
        }
    
    def _check_comment_density(self, lines: List[str]) -> Dict[str, Any]:
        """Check comment density compliance"""
        min_density = self.standards_config[StandardType.COMMENT_DENSITY].min_value
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
        
        compliant = comment_density >= min_density
        issues = []
        recommendations = []
        
        if not compliant:
            issues.append(f"Comment density {comment_density:.1f}% below minimum {min_density}%")
            issues.append(f"  Comment lines: {comment_lines}, Code lines: {code_lines}")
            recommendations.append("Add more comments to explain complex logic and functions")
        
        return {
            'compliant': compliant,
            'comment_density': comment_density,
            'comment_lines': comment_lines,
            'code_lines': code_lines,
            'issues': issues,
            'recommendations': recommendations,
            'metric': comment_density
        }
    
    def _check_module_size(self, lines: List[str]) -> Dict[str, Any]:
        """Check module size compliance"""
        max_size = self.standards_config[StandardType.MODULE_SIZE].max_value
        module_size = len(lines)
        
        compliant = module_size <= max_size
        issues = []
        recommendations = []
        
        if not compliant:
            issues.append(f"Module size {module_size} lines exceeds maximum {max_size} lines")
            recommendations.append("Split large module into smaller, focused modules")
            recommendations.append("Extract classes or functions into separate files")
        
        return {
            'compliant': compliant,
            'module_size': module_size,
            'issues': issues,
            'recommendations': recommendations,
            'metric': module_size
        }
    
    def _check_docstrings(self, file_path: Path, lines: List[str]) -> Dict[str, Any]:
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
            
            min_coverage = self.standards_config[StandardType.DOCSTRINGS].min_value
            compliant = docstring_coverage >= min_coverage
            issues = []
            recommendations = []
            
            if not compliant:
                issues.append(f"Docstring coverage {docstring_coverage:.1f}% below minimum {min_coverage}%")
                issues.append(f"  Functions: {functions_with_docstrings}/{len(functions)} with docstrings")
                issues.append(f"  Classes: {classes_with_docstrings}/{len(classes)} with docstrings")
                recommendations.append("Add docstrings to all functions and classes")
                recommendations.append("Use clear, descriptive docstrings following Google or NumPy style")
            
            return {
                'compliant': compliant,
                'docstring_coverage': docstring_coverage,
                'functions': len(functions),
                'functions_with_docstrings': functions_with_docstrings,
                'classes': len(classes),
                'classes_with_docstrings': classes_with_docstrings,
                'issues': issues,
                'recommendations': recommendations,
                'metric': docstring_coverage
            }
            
        except Exception as e:
            logger.warning(f"Could not parse {file_path} for docstring analysis: {e}")
            return {
                'compliant': False,
                'error': str(e),
                'issues': [f"Could not analyze docstrings: {e}"],
                'recommendations': ["Fix syntax errors in file"],
                'metric': 0.0
            }
    
    def _check_naming_conventions(self, file_path: Path, lines: List[str]) -> Dict[str, Any]:
        """Check Python naming convention compliance"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            violations = []
            
            # Check function names (snake_case)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        violations.append(f"Function '{node.name}' should use snake_case")
                
                # Check class names (PascalCase)
                elif isinstance(node, ast.ClassDef):
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        violations.append(f"Class '{node.name}' should use PascalCase")
                
                # Check variable names (snake_case)
                elif isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):  # Assignment
                        if not re.match(r'^[a-z_][a-z0-9_]*$', node.id) and not node.id.isupper():
                            violations.append(f"Variable '{node.id}' should use snake_case")
            
            compliant = len(violations) == 0
            issues = []
            recommendations = []
            
            if not compliant:
                issues.extend(violations[:5])  # Show first 5 violations
                if len(violations) > 5:
                    issues.append(f"... and {len(violations) - 5} more violations")
                recommendations.append("Follow Python naming conventions: snake_case for functions/variables, PascalCase for classes")
            
            return {
                'compliant': compliant,
                'violations': violations,
                'issues': issues,
                'recommendations': recommendations,
                'metric': len(violations)
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'error': str(e),
                'issues': [f"Could not analyze naming conventions: {e}"],
                'recommendations': ["Fix syntax errors in file"],
                'metric': 999
            }
    
    def _check_complexity(self, file_path: Path, lines: List[str]) -> Dict[str, Any]:
        """Check code complexity"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            complexity_issues = []
            
            # Check function complexity (simplified)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count decision points (if, for, while, etc.)
                    decision_points = 0
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                            decision_points += 1
                    
                    if decision_points > 10:  # High complexity threshold
                        complexity_issues.append(f"Function '{node.name}' has high complexity ({decision_points} decision points)")
            
            compliant = len(complexity_issues) == 0
            issues = []
            recommendations = []
            
            if not compliant:
                issues.extend(complexity_issues[:3])  # Show first 3
                if len(complexity_issues) > 3:
                    issues.append(f"... and {len(complexity_issues) - 3} more complexity issues")
                recommendations.append("Break complex functions into smaller, focused functions")
                recommendations.append("Extract complex logic into separate methods or classes")
            
            return {
                'compliant': compliant,
                'complexity_issues': complexity_issues,
                'issues': issues,
                'recommendations': recommendations,
                'metric': len(complexity_issues)
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'error': str(e),
                'issues': [f"Could not analyze complexity: {e}"],
                'recommendations': ["Fix syntax errors in file"],
                'metric': 999
            }
    
    def _calculate_compliance_score(self, standards_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall compliance score for a file"""
        if not standards_results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for standard_type, result in standards_results.items():
            config = self.standards_config[StandardType(standard_type)]
            weight = config.weight
            
            if result.get('compliant', True):
                score = 100.0
            else:
                # Calculate partial score based on how far from compliance
                if 'metric' in result:
                    metric = result['metric']
                    if config.max_value and metric > config.max_value:
                        score = max(0, 100 - (metric - config.max_value) * 2)
                    elif config.min_value and metric < config.min_value:
                        score = max(0, metric / config.min_value * 100)
                    else:
                        score = 50.0  # Default partial score
                else:
                    score = 50.0
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_priority(self, standards_results: Dict[str, Dict[str, Any]], compliance_score: float) -> str:
        """Determine priority level for file fixes"""
        if compliance_score < 30:
            return "high"
        elif compliance_score < 60:
            return "medium"
        else:
            return "low"
    
    def _get_compliance_level(self, compliance_rate: float) -> ComplianceLevel:
        """Get compliance level based on rate"""
        if compliance_rate >= 90:
            return ComplianceLevel.EXCELLENT
        elif compliance_rate >= 70:
            return ComplianceLevel.GOOD
        elif compliance_rate >= 50:
            return ComplianceLevel.FAIR
        elif compliance_rate >= 30:
            return ComplianceLevel.POOR
        else:
            return ComplianceLevel.CRITICAL
    
    def _generate_recommendations(self, file_analyses: List[FileAnalysis], standards_compliance: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall recommendations
        high_priority_files = [f for f in file_analyses if f.priority == "high"]
        if high_priority_files:
            recommendations.append(f"🔴 {len(high_priority_files)} high-priority files need immediate attention")
        
        # Standards-specific recommendations
        for standard_type, data in standards_compliance.items():
            compliance_rate = data['compliance_rate']
            if compliance_rate < 50:
                recommendations.append(f"⚠️ {standard_type.replace('_', ' ').title()} compliance critical: {compliance_rate:.1f}%")
            elif compliance_rate < 80:
                recommendations.append(f"📝 {standard_type.replace('_', ' ').title()} needs improvement: {compliance_rate:.1f}%")
        
        # File-specific recommendations
        large_files = [f for f in file_analyses if not f.standards.get('module_size', {}).get('compliant', True)]
        if large_files:
            recommendations.append(f"📦 {len(large_files)} files exceed size limits - consider splitting")
        
        files_without_docstrings = [f for f in file_analyses if not f.standards.get('docstrings', {}).get('compliant', True)]
        if files_without_docstrings:
            recommendations.append(f"📚 {len(files_without_docstrings)} files need docstring improvements")
        
        return recommendations
    
    def _calculate_progress_metrics(self, file_analyses: List[FileAnalysis], standards_compliance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate progress and improvement metrics"""
        total_files = len(file_analyses)
        high_priority = len([f for f in file_analyses if f.priority == "high"])
        medium_priority = len([f for f in file_analyses if f.priority == "medium"])
        low_priority = len([f for f in file_analyses if f.priority == "low"])
        
        avg_compliance_score = sum(f.compliance_score for f in file_analyses) / total_files if total_files > 0 else 0
        
        return {
            'priority_distribution': {
                'high': high_priority,
                'medium': medium_priority,
                'low': low_priority
            },
            'average_compliance_score': avg_compliance_score,
            'files_needing_attention': high_priority + medium_priority,
            'improvement_potential': 100 - avg_compliance_score
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
    
    def generate_detailed_report(self, analysis: RepositoryAnalysis) -> str:
        """Generate detailed markdown report"""
        report_lines = [
            "# CTAS Code Standards Analysis Report",
            "",
            f"**Repository**: {self.repo_root}",
            f"**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Compliance Level**: {analysis.compliance_level.value.upper()}",
            "",
            "## 📊 Executive Summary",
            "",
            f"- **Overall Compliance**: {analysis.summary['compliance_rate']:.1f}%",
            f"- **Total Files**: {analysis.summary['total_files']}",
            f"- **Compliant Files**: {analysis.summary['compliant_files']}",
            f"- **Non-Compliant Files**: {analysis.summary['non_compliant_files']}",
            f"- **Analysis Time**: {analysis.summary['analysis_time']:.2f}s",
            "",
            "## 📈 Standards Compliance",
            "",
            "| Standard | Compliant | Non-Compliant | Compliance Rate | Status |",
            "|----------|-----------|---------------|-----------------|--------|"
        ]
        
        for standard, data in analysis.standards.items():
            rate = data['compliance_rate']
            status = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
            report_lines.append(
                f"| {standard.replace('_', ' ').title()} | {data['compliant']} | {data['non_compliant']} | {rate:.1f}% | {status} |"
            )
        
        report_lines.extend([
            "",
            "## 🎯 Priority Actions",
            ""
        ])
        
        for recommendation in analysis.recommendations:
            report_lines.append(f"- {recommendation}")
        
        report_lines.extend([
            "",
            "## 📋 Progress Metrics",
            "",
            f"- **High Priority Files**: {analysis.progress_metrics['priority_distribution']['high']}",
            f"- **Medium Priority Files**: {analysis.progress_metrics['priority_distribution']['medium']}",
            f"- **Low Priority Files**: {analysis.progress_metrics['priority_distribution']['low']}",
            f"- **Average Compliance Score**: {analysis.progress_metrics['average_compliance_score']:.1f}%",
            f"- **Improvement Potential**: {analysis.progress_metrics['improvement_potential']:.1f}%",
            "",
            "## 🔍 Top Issues by Priority",
            ""
        ])
        
        # Show top high-priority files
        high_priority_files = [f for f in analysis.file_details if f.priority == "high"][:5]
        if high_priority_files:
            report_lines.append("### 🔴 High Priority Files")
            for file_analysis in high_priority_files:
                report_lines.append(f"- **{file_analysis.path}** ({file_analysis.compliance_score:.1f}% compliant)")
                for issue in file_analysis.issues[:2]:  # Show first 2 issues
                    report_lines.append(f"  - {issue}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_analysis(self, analysis: RepositoryAnalysis, output_path: str) -> None:
        """Save analysis results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        analysis_dict = {
            'summary': analysis.summary,
            'standards': analysis.standards,
            'file_details': [asdict(f) for f in analysis.file_details],
            'recommendations': analysis.recommendations,
            'progress_metrics': analysis.progress_metrics,
            'compliance_level': analysis.compliance_level.value
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis_dict, f, indent=2)
        
        logger.info(f"💾 Analysis saved to: {output_file}")

# Usage example
def main():
    """Example usage of enhanced CTAS code standards"""
    standards = CTASCodeStandards()
    analysis = standards.analyze_repository()
    
    print(f"\n📊 CTAS Code Standards Analysis")
    print(f"================================")
    print(f"Overall Compliance: {analysis.summary['compliance_rate']:.1f}%")
    print(f"Compliance Level: {analysis.compliance_level.value.upper()}")
    print(f"Files Analyzed: {analysis.summary['total_files']}")
    print(f"High Priority Files: {analysis.progress_metrics['priority_distribution']['high']}")
    
    # Generate detailed report
    report = standards.generate_detailed_report(analysis)
    
    # Save analysis
    standards.save_analysis(analysis, "ctas_code_standards_analysis.json")
    
    return analysis

if __name__ == "__main__":
    main() 