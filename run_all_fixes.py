#!/usr/bin/env python3
"""
Master Fix Runner - All Teams
==============================

Runs all team fixes in sequence
"""

import subprocess
import sys


def run_team_fixes():
    """Run all team fixes"""
    teams = [
        ("Team A", "team_packages/team_a_critical_fixes/critical_fixes_implementation.py"),
        ("Team B", "team_packages/team_b_large_files/large_files_fix.py"),
        ("Team C", "team_packages/team_c_database/database_optimization.py"),
        ("Team D", "team_packages/team_d_organization/code_standards.py")
    ]
    
    results = {}
    
    for team_name, script_path in teams:
        print(f"\n🚀 Running {team_name} fixes...")
        try:
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {team_name} completed successfully")
                results[team_name] = True
            else:
                print(f"❌ {team_name} failed: {result.stderr}")
                results[team_name] = False
        except Exception as e:
            print(f"❌ {team_name} error: {e}")
            results[team_name] = False
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n🎯 Final Summary: {successful}/{total} teams completed successfully")
    
    if successful == total:
        print("🎉 All teams completed successfully!")
        return True
    else:
        print("⚠️ Some teams had issues")
        return False


if __name__ == "__main__":
    success = run_team_fixes()
    sys.exit(0 if success else 1)
