"""
OPSEC Manager Module
------------------
This module provides operational security management for the NyxTrace platform,
including browser fingerprint protection, network security, and identity protection.
"""

import os
import json
import logging
import requests
import socket
import subprocess
import platform
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from enum import Enum
import hashlib
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("opsec_manager")


class ProtectionLevel(Enum):
    """Enumeration of protection levels"""

    OFF = "off"
    BASIC = "basic"
    STANDARD = "standard"
    MAXIMUM = "maximum"


class OperationalProfile:
    """Class representing an operational security profile"""

    def __init__(
        self,
        name: str,
        description: str,
        browser_protection: ProtectionLevel = ProtectionLevel.STANDARD,
        network_protection: ProtectionLevel = ProtectionLevel.STANDARD,
        identity_protection: ProtectionLevel = ProtectionLevel.STANDARD,
    ):
        """
        Initialize an operational profile

        Args:
            name: Profile name
            description: Profile description
            browser_protection: Browser protection level
            network_protection: Network protection level
            identity_protection: Identity protection level
        """
        self.name = name
        self.description = description
        self.browser_protection = browser_protection
        self.network_protection = network_protection
        self.identity_protection = identity_protection
        self.settings = {
            "browser": {
                "user_agent_randomization": browser_protection
                != ProtectionLevel.OFF,
                "webrtc_protection": browser_protection != ProtectionLevel.OFF,
                "canvas_fingerprint_protection": browser_protection
                in [ProtectionLevel.STANDARD, ProtectionLevel.MAXIMUM],
                "audio_fingerprint_protection": browser_protection
                == ProtectionLevel.MAXIMUM,
                "hardware_fingerprint_protection": browser_protection
                == ProtectionLevel.MAXIMUM,
            },
            "network": {
                "vpn_enabled": network_protection != ProtectionLevel.OFF,
                "proxy_enabled": network_protection
                in [ProtectionLevel.STANDARD, ProtectionLevel.MAXIMUM],
                "dns_protection": network_protection != ProtectionLevel.OFF,
                "traffic_obfuscation": network_protection
                == ProtectionLevel.MAXIMUM,
                "bandwidth_management": network_protection
                == ProtectionLevel.MAXIMUM,
            },
            "identity": {
                "disable_browser_history": identity_protection
                != ProtectionLevel.OFF,
                "clear_cookies_on_exit": identity_protection
                != ProtectionLevel.OFF,
                "advanced_tracking_protection": identity_protection
                in [ProtectionLevel.STANDARD, ProtectionLevel.MAXIMUM],
                "isolated_sessions": identity_protection
                == ProtectionLevel.MAXIMUM,
                "location_masking": identity_protection
                == ProtectionLevel.MAXIMUM,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary

        Returns:
            Dictionary representation of the profile
        """
        return {
            "name": self.name,
            "description": self.description,
            "browser_protection": self.browser_protection.value,
            "network_protection": self.network_protection.value,
            "identity_protection": self.identity_protection.value,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperationalProfile":
        """
        Create profile from dictionary

        Args:
            data: Dictionary representation of profile

        Returns:
            OperationalProfile instance
        """
        profile = cls(
            name=data["name"],
            description=data["description"],
            browser_protection=ProtectionLevel(data["browser_protection"]),
            network_protection=ProtectionLevel(data["network_protection"]),
            identity_protection=ProtectionLevel(data["identity_protection"]),
        )

        # Update settings
        if "settings" in data:
            profile.settings = data["settings"]

        return profile


class OpsecManager:
    """
    Main class for managing operational security

    This class provides methods for:
    - Managing operational profiles
    - Checking system security status
    - Implementing security measures
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the OPSEC Manager

        Args:
            config_path: Optional path to configuration file
        """
        self.profiles = {}
        self.active_profile = None
        self.security_status = {}
        self.config_path = config_path or os.path.join(
            os.getcwd(), "config", "opsec.json"
        )

        # Create default profiles
        self._create_default_profiles()

        # Load configuration
        self._load_config()

        # Activate default profile if none active
        if not self.active_profile:
            self.activate_profile("Default Profile")

        logger.info("OPSEC Manager initialized")

    def _create_default_profiles(self) -> None:
        """Create default operational profiles"""
        # Default Profile
        default_profile = OperationalProfile(
            name="Default Profile",
            description="Standard operational profile with basic security measures.",
            browser_protection=ProtectionLevel.BASIC,
            network_protection=ProtectionLevel.BASIC,
            identity_protection=ProtectionLevel.BASIC,
        )
        self.profiles[default_profile.name] = default_profile

        # Anonymous Profile
        anonymous_profile = OperationalProfile(
            name="Anonymous Profile",
            description="High-security profile designed for anonymous operations.",
            browser_protection=ProtectionLevel.STANDARD,
            network_protection=ProtectionLevel.STANDARD,
            identity_protection=ProtectionLevel.STANDARD,
        )
        self.profiles[anonymous_profile.name] = anonymous_profile

        # Field Operations Profile
        field_profile = OperationalProfile(
            name="Field Operations",
            description="Specialized profile for mobile/field operations with reduced footprint.",
            browser_protection=ProtectionLevel.MAXIMUM,
            network_protection=ProtectionLevel.MAXIMUM,
            identity_protection=ProtectionLevel.MAXIMUM,
        )
        self.profiles[field_profile.name] = field_profile

    def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)

                # Load profiles
                if "profiles" in config:
                    for profile_data in config["profiles"]:
                        profile = OperationalProfile.from_dict(profile_data)
                        self.profiles[profile.name] = profile

                # Set active profile
                if (
                    "active_profile" in config
                    and config["active_profile"] in self.profiles
                ):
                    self.active_profile = self.profiles[
                        config["active_profile"]
                    ]

                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.info("No configuration file found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def _save_config(self) -> None:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            config = {
                "profiles": [
                    profile.to_dict() for profile in self.profiles.values()
                ],
                "active_profile": (
                    self.active_profile.name if self.active_profile else None
                ),
            }

            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")

    def get_profiles(self) -> List[Dict[str, Any]]:
        """
        Get available operational profiles

        Returns:
            List of profile dictionaries
        """
        return [profile.to_dict() for profile in self.profiles.values()]

    def get_active_profile(self) -> Optional[Dict[str, Any]]:
        """
        Get the active operational profile

        Returns:
            Active profile dictionary, or None if no profile is active
        """
        return self.active_profile.to_dict() if self.active_profile else None

    def create_profile(
        self,
        name: str,
        description: str,
        browser_protection: Union[
            ProtectionLevel, str
        ] = ProtectionLevel.STANDARD,
        network_protection: Union[
            ProtectionLevel, str
        ] = ProtectionLevel.STANDARD,
        identity_protection: Union[
            ProtectionLevel, str
        ] = ProtectionLevel.STANDARD,
    ) -> bool:
        """
        Create a new operational profile

        Args:
            name: Profile name
            description: Profile description
            browser_protection: Browser protection level
            network_protection: Network protection level
            identity_protection: Identity protection level

        Returns:
            True if profile created, False if profile already exists
        """
        if name in self.profiles:
            logger.warning(f"Profile {name} already exists")
            return False

        # Convert string enums to enum types if necessary
        if isinstance(browser_protection, str):
            browser_protection = ProtectionLevel(browser_protection)

        if isinstance(network_protection, str):
            network_protection = ProtectionLevel(network_protection)

        if isinstance(identity_protection, str):
            identity_protection = ProtectionLevel(identity_protection)

        # Create profile
        profile = OperationalProfile(
            name=name,
            description=description,
            browser_protection=browser_protection,
            network_protection=network_protection,
            identity_protection=identity_protection,
        )

        # Add to profiles
        self.profiles[name] = profile

        # Save configuration
        self._save_config()

        logger.info(f"Created profile {name}")

        return True

    def update_profile(
        self,
        name: str,
        description: Optional[str] = None,
        browser_protection: Optional[Union[ProtectionLevel, str]] = None,
        network_protection: Optional[Union[ProtectionLevel, str]] = None,
        identity_protection: Optional[Union[ProtectionLevel, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an operational profile

        Args:
            name: Profile name
            description: Optional new description
            browser_protection: Optional new browser protection level
            network_protection: Optional new network protection level
            identity_protection: Optional new identity protection level
            settings: Optional new settings

        Returns:
            True if profile updated, False if profile not found
        """
        if name not in self.profiles:
            logger.warning(f"Profile {name} not found")
            return False

        profile = self.profiles[name]

        # Update description
        if description:
            profile.description = description

        # Update browser protection
        if browser_protection:
            if isinstance(browser_protection, str):
                browser_protection = ProtectionLevel(browser_protection)
            profile.browser_protection = browser_protection

        # Update network protection
        if network_protection:
            if isinstance(network_protection, str):
                network_protection = ProtectionLevel(network_protection)
            profile.network_protection = network_protection

        # Update identity protection
        if identity_protection:
            if isinstance(identity_protection, str):
                identity_protection = ProtectionLevel(identity_protection)
            profile.identity_protection = identity_protection

        # Update settings
        if settings:
            # Merge settings
            for category, category_settings in settings.items():
                if category in profile.settings:
                    profile.settings[category].update(category_settings)

        # Save configuration
        self._save_config()

        # Update active profile if this is the active profile
        if self.active_profile and self.active_profile.name == name:
            self.active_profile = profile
            self._apply_profile_settings()

        logger.info(f"Updated profile {name}")

        return True

    def delete_profile(self, name: str) -> bool:
        """
        Delete an operational profile

        Args:
            name: Profile name

        Returns:
            True if profile deleted, False if profile not found or is active
        """
        if name not in self.profiles:
            logger.warning(f"Profile {name} not found")
            return False

        # Cannot delete active profile
        if self.active_profile and self.active_profile.name == name:
            logger.warning(f"Cannot delete active profile {name}")
            return False

        # Delete profile
        del self.profiles[name]

        # Save configuration
        self._save_config()

        logger.info(f"Deleted profile {name}")

        return True

    def activate_profile(self, name: str) -> bool:
        """
        Activate an operational profile

        Args:
            name: Profile name

        Returns:
            True if profile activated, False if profile not found
        """
        if name not in self.profiles:
            logger.warning(f"Profile {name} not found")
            return False

        # Set active profile
        self.active_profile = self.profiles[name]

        # Apply profile settings
        self._apply_profile_settings()

        # Save configuration
        self._save_config()

        logger.info(f"Activated profile {name}")

        return True

    def _apply_profile_settings(self) -> None:
        """Apply settings from active profile"""
        if not self.active_profile:
            logger.warning("No active profile")
            return

        # This would apply the actual settings to the system
        # For now, we'll just log the settings
        logger.info(
            f"Applying settings from profile {self.active_profile.name}"
        )
        logger.info(
            f"Browser protection: {self.active_profile.browser_protection.value}"
        )
        logger.info(
            f"Network protection: {self.active_profile.network_protection.value}"
        )
        logger.info(
            f"Identity protection: {self.active_profile.identity_protection.value}"
        )

    def get_system_security_status(self) -> Dict[str, Any]:
        """
        Get the current system security status

        Returns:
            Dictionary with security status
        """
        # Update security status
        self._check_security_status()

        return self.security_status

    def _check_security_status(self) -> None:
        """Check and update system security status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "os": platform.system(),
                "os_release": platform.release(),
                "hostname": socket.gethostname(),
                "machine_id": self._get_machine_id(),
            },
            "network": {
                "ip_address": self._get_ip_address(),
                "dns_servers": self._get_dns_servers(),
                "vpn_status": self._check_vpn_status(),
                "proxy_status": self._check_proxy_status(),
            },
            "browser": {
                "user_agent": self._get_user_agent(),
                "webrtc_leak": self._check_webrtc_leak(),
                "canvas_fingerprint": self._check_canvas_fingerprint(),
                "audio_fingerprint": self._check_audio_fingerprint(),
            },
            "checks": {
                "webrtc_leak_test": not self._check_webrtc_leak(),
                "dns_leak_test": not self._check_dns_leak(),
                "browser_fingerprint_test": not self._check_canvas_fingerprint(),
                "tracking_protection_test": self._check_tracking_protection(),
                "javascript_analysis": not self._check_javascript_enabled(),
                "cookie_analysis": self._check_cookie_status(),
            },
        }

        self.security_status = status

    def _get_machine_id(self) -> str:
        """Get machine ID"""
        # This is a placeholder - in a real system, this would get a unique identifier
        return hashlib.md5(socket.gethostname().encode()).hexdigest()

    def _get_ip_address(self) -> str:
        """Get IP address"""
        try:
            # This gets the IP address the system uses to connect to the internet
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _get_dns_servers(self) -> List[str]:
        """Get DNS servers"""
        # This is a placeholder - in a real system, this would get the actual DNS servers
        return ["1.1.1.1", "8.8.8.8"]

    def _check_vpn_status(self) -> bool:
        """Check if VPN is active"""
        # This is a placeholder - in a real system, this would check if a VPN is active
        return True

    def _check_proxy_status(self) -> bool:
        """Check if proxy is active"""
        # This is a placeholder - in a real system, this would check if a proxy is active
        return True

    def _get_user_agent(self) -> str:
        """Get browser user agent"""
        # This is a placeholder - in a real system, this would get the actual user agent
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    def _check_webrtc_leak(self) -> bool:
        """Check for WebRTC leaks"""
        # This is a placeholder - in a real system, this would check for WebRTC leaks
        return False

    def _check_canvas_fingerprint(self) -> bool:
        """Check if canvas fingerprinting is detectable"""
        # This is a placeholder - in a real system, this would check for canvas fingerprinting
        return False

    def _check_audio_fingerprint(self) -> bool:
        """Check if audio fingerprinting is detectable"""
        # This is a placeholder - in a real system, this would check for audio fingerprinting
        return False

    def _check_dns_leak(self) -> bool:
        """Check for DNS leaks"""
        # This is a placeholder - in a real system, this would check for DNS leaks
        return False

    def _check_tracking_protection(self) -> bool:
        """Check if tracking protection is working"""
        # This is a placeholder - in a real system, this would check if tracking protection is working
        return True

    def _check_javascript_enabled(self) -> bool:
        """Check if JavaScript is enabled"""
        # This is a placeholder - in a real system, this would check if JavaScript is enabled
        return True

    def _check_cookie_status(self) -> Dict[str, Any]:
        """Check cookie status"""
        # This is a placeholder - in a real system, this would check cookie status
        return {"total": 25, "third_party": 15, "tracking": 10}

    def generate_random_user_agent(self) -> str:
        """
        Generate a random user agent string

        Returns:
            Random user agent string
        """
        # This is a placeholder - in a real system, this would generate a random user agent
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        ]

        return random.choice(user_agents)

    def get_opsec_guidelines(self) -> Dict[str, List[str]]:
        """
        Get OPSEC guidelines

        Returns:
            Dictionary with guidelines by category
        """
        return {
            "browser": [
                "Use a dedicated browser profile for each investigation",
                "Disable JavaScript when possible",
                "Use HTTPS-only mode",
                "Disable browser fingerprinting",
                "Clear cookies and cache between sessions",
            ],
            "network": [
                "Use a VPN or Tor for sensitive investigations",
                "Consider using a dedicated device on a separate network",
                "Regularly change IP addresses during long investigations",
                "Use time delays between requests to avoid detection",
            ],
            "operational": [
                "Maintain separation between investigation personas",
                "Never use personal accounts or identifiers",
                "Document the investigation methodology",
                "Use secure storage for findings",
                "Be aware of legal and ethical boundaries",
            ],
            "tools": [
                "Use virtual machines or containers for isolation",
                "Consider using specialized OSINT frameworks",
                "Maintain separate environments for different investigation types",
                "Use encryption for storing investigation results",
            ],
        }
