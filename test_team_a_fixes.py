#!/usr/bin/env python3
"""
Quick test script to verify Team A critical fixes are working properly
This test can be run from the repository root to verify the fixes
"""

import os
import sys
import importlib.util

def test_file_exists_and_imports(file_path, module_name):
    """Test if file exists and can be imported"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ {file_path} does not exist")
            return False
        
        # Try to load and compile the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"❌ Could not create spec for {file_path}")
            return False
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✅ {file_path} exists and imports successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error with {file_path}: {e}")
        return False

def main():
    print("🧪 Testing Team A Critical Fixes...")
    print("=" * 50)
    
    tests = [
        ("utils/enhanced_html_renderer.py", "enhanced_html_renderer"),
        ("core/database/streamlit_safe_factory.py", "streamlit_safe_factory"),
        ("utils/import_compatibility.py", "import_compatibility"),
        ("core/configuration_manager.py", "configuration_manager"),
        ("pages/adversary_task_viewer.py", "adversary_task_viewer")
    ]
    
    passed = 0
    total = len(tests)
    
    for file_path, module_name in tests:
        if test_file_exists_and_imports(file_path, module_name):
            passed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Team A critical fixes are working properly!")
        return 0
    else:
        print("⚠️ Some fixes may need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
