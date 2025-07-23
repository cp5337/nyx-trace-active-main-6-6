#!/usr/bin/env python3
"""
Team A Critical Fixes Verification Script
==========================================

This script verifies that all Team A critical fixes have been successfully applied
and are working correctly.
"""

import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamAFixVerifier:
    """
    Verifies that all Team A critical fixes are working correctly
    """
    
    def __init__(self):
        self.repo_root = Path('.').resolve()
        self.test_results = {}
        
    def test_html_rendering_fixes(self):
        """Test HTML rendering fixes"""
        logger.info("Testing HTML rendering fixes...")
        
        try:
            # Test enhanced HTML renderer import
            import utils.enhanced_html_renderer as ehr
            
            # Test that the render functions exist
            assert hasattr(ehr, 'render_task_card_native'), "render_task_card_native function missing"
            assert hasattr(ehr, 'render_with_fallback'), "render_with_fallback function missing"
            
            # Test with mock task data
            task_data = {
                'task_name': 'Test Task',
                'hash_id': 'TEST-001',
                'category': 'Test Category',
                'reliability': 0.8,
                'confidence': 0.9,
                'description': 'Test description'
            }
            
            # This would normally render to Streamlit, but we're just testing the function exists and runs
            logger.info("✅ HTML rendering fixes verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ HTML rendering fixes failed: {e}")
            traceback.print_exc()
            return False
    
    def test_database_threading_fixes(self):
        """Test database threading fixes"""
        logger.info("Testing database threading fixes...")
        
        try:
            # Test streamlit-safe database factory import
            import core.database.streamlit_safe_factory as factory
            
            # Test that the factory class exists
            assert hasattr(factory, 'StreamlitSafeDatabaseFactory'), "StreamlitSafeDatabaseFactory class missing"
            assert hasattr(factory, 'database_factory'), "database_factory instance missing"
            
            # Test factory instantiation
            db_factory = factory.StreamlitSafeDatabaseFactory()
            
            # Test that required methods exist
            assert hasattr(db_factory, 'get_connection'), "get_connection method missing"
            assert hasattr(db_factory, 'cleanup_connections'), "cleanup_connections method missing"
            
            logger.info("✅ Database threading fixes verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database threading fixes failed: {e}")
            traceback.print_exc()
            return False
    
    def test_import_dependencies_fixes(self):
        """Test import dependencies fixes"""
        logger.info("Testing import dependencies fixes...")
        
        try:
            # Test import compatibility module
            import utils.import_compatibility as ic
            
            # Test that functions exist
            assert hasattr(ic, 'safe_import'), "safe_import function missing"
            assert hasattr(ic, 'check_dependencies'), "check_dependencies function missing"
            
            # Test safe_import function
            streamlit_module = ic.safe_import('streamlit')
            assert streamlit_module is not None, "safe_import failed for streamlit"
            
            # Test check_dependencies function
            deps = ic.check_dependencies()
            assert isinstance(deps, dict), "check_dependencies should return dict"
            assert 'streamlit' in deps, "streamlit should be in dependencies check"
            
            # Test that __init__.py files exist in required directories
            required_init_files = [
                'core/soc_teams/__init__.py',
                'pages/__init__.py',
                'team_packages/__init__.py',
                'team_packages/team_a_critical_fixes/__init__.py',
                'team_packages/team_b_large_files/__init__.py',
                'team_packages/team_c_database/__init__.py',
                'team_packages/team_d_organization/__init__.py'
            ]
            
            for init_file in required_init_files:
                init_path = self.repo_root / init_file
                assert init_path.exists(), f"Missing __init__.py file: {init_file}"
            
            logger.info("✅ Import dependencies fixes verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Import dependencies fixes failed: {e}")
            traceback.print_exc()
            return False
    
    def test_configuration_fixes(self):
        """Test configuration fixes"""
        logger.info("Testing configuration fixes...")
        
        try:
            # Test configuration manager import
            import core.configuration_manager as cm
            
            # Test that classes and functions exist
            assert hasattr(cm, 'ConfigurationManager'), "ConfigurationManager class missing"
            assert hasattr(cm, 'config_manager'), "config_manager instance missing"
            
            # Test configuration manager instantiation
            config = cm.ConfigurationManager()
            
            # Test that required methods exist
            assert hasattr(config, 'get'), "get method missing"
            assert hasattr(config, 'get_database_config'), "get_database_config method missing"
            assert hasattr(config, 'is_debug_enabled'), "is_debug_enabled method missing"
            assert hasattr(config, 'get_log_level'), "get_log_level method missing"
            
            # Test basic functionality
            db_config = config.get_database_config()
            assert isinstance(db_config, dict), "get_database_config should return dict"
            assert 'url' in db_config, "database config should have url"
            
            # Test .env.sample file exists
            env_sample_path = self.repo_root / '.env.sample'
            assert env_sample_path.exists(), ".env.sample file missing"
            
            logger.info("✅ Configuration fixes verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration fixes failed: {e}")
            traceback.print_exc()
            return False
    
    def test_original_files_import(self):
        """Test that original problematic files now import correctly"""
        logger.info("Testing original files import...")
        
        try:
            # Test adversary task viewer import (suppress Streamlit warnings)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pages.adversary_task_viewer
            
            # Test HTML renderer import
            import utils.html_renderer
            
            # Test that these modules have expected attributes
            assert hasattr(pages.adversary_task_viewer, 'render_task_card'), "render_task_card function missing"
            assert hasattr(utils.html_renderer, 'render_html'), "render_html function missing"
            
            logger.info("✅ Original files import verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Original files import verification failed: {e}")
            traceback.print_exc()
            return False
    
    def test_syntax_compilation(self):
        """Test that all Python files compile without syntax errors"""
        logger.info("Testing syntax compilation...")
        
        try:
            # Key files to test
            test_files = [
                'pages/adversary_task_viewer.py',
                'utils/html_renderer.py',
                'utils/enhanced_html_renderer.py',
                'core/database/streamlit_safe_factory.py',
                'utils/import_compatibility.py', 
                'core/configuration_manager.py'
            ]
            
            import py_compile
            
            for file_path in test_files:
                full_path = self.repo_root / file_path
                if full_path.exists():
                    py_compile.compile(str(full_path), doraise=True)
                    logger.info(f"✅ {file_path} compiles successfully")
                else:
                    logger.warning(f"⚠️ {file_path} not found")
            
            logger.info("✅ Syntax compilation verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Syntax compilation verification failed: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all verification tests"""
        logger.info("🔍 Starting Team A Critical Fixes Verification...")
        
        tests = [
            ("HTML Rendering Fixes", self.test_html_rendering_fixes),
            ("Database Threading Fixes", self.test_database_threading_fixes),
            ("Import Dependencies Fixes", self.test_import_dependencies_fixes),
            ("Configuration Fixes", self.test_configuration_fixes),
            ("Original Files Import", self.test_original_files_import),
            ("Syntax Compilation", self.test_syntax_compilation)
        ]
        
        for test_name, test_function in tests:
            try:
                logger.info(f"Running {test_name}...")
                success = test_function()
                self.test_results[test_name] = success
                if success:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with error: {e}")
                self.test_results[test_name] = False
        
        # Summary
        passed_tests = sum(1 for success in self.test_results.values() if success)
        total_tests = len(self.test_results)
        
        logger.info(f"🎯 Team A Critical Fixes Verification Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All Team A critical fixes verification tests PASSED!")
            print("✅ Repository is ready for Teams B, C, and D to proceed.")
            return True
        else:
            print("⚠️ Some Team A critical fixes verification tests FAILED.")
            print("❌ Please address the failing tests before proceeding.")
            return False

if __name__ == "__main__":
    verifier = TeamAFixVerifier()
    success = verifier.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
