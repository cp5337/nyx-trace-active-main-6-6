#!/usr/bin/env python3
"""
Code Standards Cleanup Script
=============================

This script fixes common code style issues in the Team A critical fixes
to meet Python PEP 8 standards.
"""

import os
import re
import sys
from pathlib import Path


class CodeStandardsCleanup:
    """
    Cleans up common code style issues
    """
    
    def __init__(self, repo_root=None):
        self.repo_root = Path(repo_root) if repo_root else Path('.').resolve()
        
    def clean_whitespace(self, file_path):
        """Remove trailing whitespace and fix blank line issues"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for line in lines:
            # Remove trailing whitespace
            cleaned_line = line.rstrip() + '\n'
            cleaned_lines.append(cleaned_line)
        
        # Ensure file ends with newline
        if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
            cleaned_lines[-1] += '\n'
        
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
    
    def fix_imports(self, file_path):
        """Fix import issues"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove unused imports (basic patterns)
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                if in_imports:
                    import_lines.append(line)
                else:
                    other_lines.append(line)
            elif line.strip() == '' and in_imports:
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Remove known unused imports
        filtered_imports = []
        for line in import_lines:
            # Skip known unused imports
            skip_patterns = [
                'import sys',
                'from typing import Dict, Any, Optional',
                'from pathlib import Path'
            ]
            
            should_skip = False
            for pattern in skip_patterns:
                if pattern in line and any(
                    unused in line for unused in ['sys', 'Dict', 'Any', 'Optional', 'Path']
                ):
                    # Only skip if these are the only imports or clearly unused
                    if (line.strip() == pattern or 
                        ('sys' in line and 'sys' not in '\n'.join(other_lines)) or
                        ('Path' in line and 'Path' not in '\n'.join(other_lines))):
                        should_skip = True
                        break
            
            if not should_skip:
                filtered_imports.append(line)
        
        # Reconstruct content
        new_content = '\n'.join(filtered_imports + other_lines)
        
        with open(file_path, 'w') as f:
            f.write(new_content)
    
    def fix_blank_lines(self, file_path):
        """Fix blank line spacing issues"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        prev_line = ""
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            prev_stripped = prev_line.strip()
            
            # Add proper spacing before function definitions
            if (stripped.startswith('def ') and 
                prev_stripped and 
                not prev_stripped.startswith('"""') and
                not prev_stripped.endswith('"""')):
                # Check if we need 2 blank lines
                if i > 0 and not prev_stripped == '':
                    # Count existing blank lines
                    blank_count = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    if blank_count < 2:
                        # Add missing blank lines
                        for _ in range(2 - blank_count):
                            new_lines.append('\n')
            
            # Add proper spacing before class definitions  
            if (stripped.startswith('class ') and 
                prev_stripped and 
                not prev_stripped.startswith('"""') and
                not prev_stripped.endswith('"""')):
                if i > 0 and not prev_stripped == '':
                    blank_count = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    if blank_count < 2:
                        for _ in range(2 - blank_count):
                            new_lines.append('\n')
            
            new_lines.append(line)
            prev_line = line
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
    
    def clean_file(self, file_path):
        """Clean a single file"""
        print(f"Cleaning {file_path}...")
        
        try:
            self.clean_whitespace(file_path)
            self.fix_blank_lines(file_path)
            print(f"✅ Cleaned {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error cleaning {file_path}: {e}")
            return False
    
    def clean_all_files(self):
        """Clean all Team A critical fixes files"""
        files_to_clean = [
            'utils/enhanced_html_renderer.py',
            'core/database/streamlit_safe_factory.py', 
            'utils/import_compatibility.py',
            'core/configuration_manager.py'
        ]
        
        results = {}
        for file_path in files_to_clean:
            full_path = self.repo_root / file_path
            if full_path.exists():
                results[file_path] = self.clean_file(full_path)
            else:
                print(f"⚠️ File not found: {file_path}")
                results[file_path] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\n🎯 Code Standards Cleanup Summary: {successful}/{total} files cleaned")
        
        return successful == total


if __name__ == "__main__":
    cleaner = CodeStandardsCleanup()
    success = cleaner.clean_all_files()
    
    if success:
        print("\n🎉 All files cleaned successfully!")
        print("✅ Code standards compliance improved.")
    else:
        print("\n⚠️ Some files could not be cleaned.")
        print("❌ Manual review may be required.")
    
    sys.exit(0 if success else 1)
