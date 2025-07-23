#!/usr/bin/env python3
"""
Team D Organization Management Implementation
===========================================

This module provides the necessary infrastructure for team organization and 
role management within the nyx-trace project.

Key Features:
- Team and role definition
- Access management
- Team collaboration tools

"""

import json
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrganizationManager:
    """
    Organization and Role Management System
    """
    
    def __init__(self, roles_file='roles.json'):
        self.roles_file = roles_file
        self.roles = {}
        
        self.load_roles()
        logger.info("Organization Manager initialized.")
    
    def load_roles(self):
        """Load roles from JSON file."""
        try:
            with open(self.roles_file, 'r') as f:
                self.roles = json.load(f)
                logger.info(f"Loaded roles from {self.roles_file}.")
        except FileNotFoundError:
            logger.warning(f"Roles file {self.roles_file} not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON in roles file: {e}")
        
    def save_roles(self):
        """Save roles to JSON file."""
        try:
            with open(self.roles_file, 'w') as f:
                json.dump(self.roles, f, indent=2)
                logger.info(f"Roles saved to {self.roles_file}.")
        except Exception as e:
            logger.error(f"Failed to save roles: {e}")
    
    def define_role(self, role_name: str, permissions: List[str]) -> bool:
        """Define a new role with specified permissions."""
        if role_name in self.roles:
            logger.warning(f"Role '{role_name}' already exists.")
            return False
        
        self.roles[role_name] = {
            'permissions': permissions,
            'members': []
        }
        
        logger.info(f"Defined new role: {role_name}.")
        self.save_roles()
        return True
    
    def assign_member_to_role(self, member_name: str, role_name: str) -> bool:
        """Assign a member to a specific role."""
        if role_name not in self.roles:
            logger.error(f"Role '{role_name}' does not exist.")
            return False
        
        if member_name in self.roles[role_name]['members']:
            logger.warning(f"Member '{member_name}' is already in role '{role_name}'.")
            return False

        self.roles[role_name]['members'].append(member_name)
        logger.info(f"Added member '{member_name}' to role '{role_name}'.")
        self.save_roles()
        return True
    
    def remove_member_from_role(self, member_name: str, role_name: str) -> bool:
        """Remove a member from a specific role."""
        if role_name not in self.roles:
            logger.error(f"Role '{role_name}' does not exist.")
            return False
        
        if member_name not in self.roles[role_name]['members']:
            logger.warning(f"Member '{member_name}' is not in role '{role_name}'.")
            return False

        self.roles[role_name]['members'].remove(member_name)
        logger.info(f"Removed member '{member_name}' from role '{role_name}'.")
        self.save_roles()
        return True
    
    def list_roles(self) -> List[str]:
        """List all defined roles."""
        return list(self.roles.keys())

if __name__ == "__main__":
    org_manager = OrganizationManager()
    org_manager.define_role("admin", ["manage_users", "manage_system"])
    org_manager.assign_member_to_role("Alice", "admin")
    roles = org_manager.list_roles()
    print("Defined roles:", roles)
