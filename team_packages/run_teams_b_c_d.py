#!/usr/bin/env python3
"""
Teams B, C, and D Integration Script
==================================

This script runs all Team B, C, and D implementations in sequence:
- Team B: Large Files Management
- Team C: Database Management
- Team D: Organization Management

Usage: python3 run_teams_b_c_d.py
"""

import sys
import logging
from pathlib import Path

# Add team packages to path
sys.path.append(str(Path(__file__).parent))

from team_b_large_files.large_files_manager import LargeFilesManager
from team_c_database.database_manager import DatabaseManager
from team_d_organization.organization_manager import OrganizationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_team_b():
    """Run Team B Large Files Management"""
    logger.info("🚀 Starting Team B: Large Files Management")
    
    try:
        manager = LargeFilesManager()
        results = manager.run_full_analysis(dry_run=True)
        
        if results:
            logger.info(f"✅ Team B completed successfully!")
            logger.info(f"   Found {len(results['large_files'])} large files")
            logger.info(f"   Total size: {results['analysis']['total_size_mb']}MB")
            return True
        else:
            logger.info("✅ Team B: No large files found - repository is optimized")
            return True
            
    except Exception as e:
        logger.error(f"❌ Team B failed: {e}")
        return False

def run_team_c():
    """Run Team C Database Management"""
    logger.info("🚀 Starting Team C: Database Management")
    
    try:
        db_manager = DatabaseManager("test_nyx_trace.db")
        
        # Test database operations
        db_manager.execute_query("CREATE TABLE IF NOT EXISTS team_c_test (id INTEGER PRIMARY KEY, name TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        db_manager.execute_query("INSERT INTO team_c_test (name) VALUES (?)", ["Team C Test"])
        results = db_manager.execute_query("SELECT COUNT(*) as count FROM team_c_test")
        
        count = results[0]['count'] if results else 0
        logger.info(f"✅ Team C completed successfully!")
        logger.info(f"   Database operations tested with {count} records")
        
        db_manager.close_all_connections()
        return True
        
    except Exception as e:
        logger.error(f"❌ Team C failed: {e}")
        return False

def run_team_d():
    """Run Team D Organization Management"""
    logger.info("🚀 Starting Team D: Organization Management")
    
    try:
        org_manager = OrganizationManager("team_d_roles.json")
        
        # Set up organizational structure
        org_manager.define_role("team_lead", ["manage_team", "review_code", "assign_tasks"])
        org_manager.define_role("developer", ["write_code", "run_tests", "create_docs"])
        org_manager.define_role("analyst", ["analyze_data", "create_reports", "review_security"])
        
        # Assign team members
        org_manager.assign_member_to_role("Alice", "team_lead")
        org_manager.assign_member_to_role("Bob", "developer")
        org_manager.assign_member_to_role("Charlie", "analyst")
        org_manager.assign_member_to_role("Diana", "developer")
        
        roles = org_manager.list_roles()
        logger.info(f"✅ Team D completed successfully!")
        logger.info(f"   Created {len(roles)} organizational roles: {', '.join(roles)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Team D failed: {e}")
        return False

def main():
    """Main execution function"""
    logger.info("🎯 Starting Teams B, C, and D Integration")
    logger.info("=" * 60)
    
    results = {}
    
    # Run Team B
    results['team_b'] = run_team_b()
    logger.info("-" * 40)
    
    # Run Team C
    results['team_c'] = run_team_c()
    logger.info("-" * 40)
    
    # Run Team D
    results['team_d'] = run_team_d()
    logger.info("-" * 40)
    
    # Summary
    successful_teams = sum(1 for success in results.values() if success)
    total_teams = len(results)
    
    logger.info("=" * 60)
    logger.info(f"🎯 Teams B, C, D Integration Summary: {successful_teams}/{total_teams} successful")
    
    for team, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {team.upper()}: {status}")
    
    if successful_teams == total_teams:
        logger.info("🎉 All teams completed successfully!")
        return 0
    else:
        logger.warning("⚠️ Some teams encountered issues. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
