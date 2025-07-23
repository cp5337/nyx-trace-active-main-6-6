"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-CYBERWARFARE-PARSERS-0001      │
// │ 📁 domain       : Cyberwarfare, Offensive Security, Analysis│
// │ 🧠 description  : Cyberwarfare results parser module        │
// │                  Parses output from various security tools  │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CYBERWARFARE                        │
// │ 🧩 dependencies : re, json, xml.etree                      │
// │ 🔧 tool_usage   : Analysis, Parsing                        │
// │ 📡 input_type   : Tool output text, XML, JSON              │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : parsing, extraction, standardization     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Cyberwarfare Results Parser Module
----------------------------------
This module provides parsers for the output of various security tools used
in offensive security operations. It converts raw tool output into structured
data formats for analysis and reporting.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

# Setup module logger
logger = logging.getLogger(__name__)


def parse_nmap_results(output: str) -> List[Dict[str, Any]]:
    """
    Parse nmap scan results from XML or text output

    # Function parses subject results
    # Method extracts predicate data
    # Operation processes object text

    Args:
        output: nmap output (XML or text)

    Returns:
        List of parsed findings
    """
    results = []

    # Function checks subject format
    # Method detects predicate type
    # Condition tests object content
    if output.strip().startswith("<?xml"):
        try:
            # Function parses subject XML
            # Method processes predicate format
            # Operation extracts object data
            return _parse_nmap_xml(output)
        except Exception as e:
            logger.error(f"Error parsing nmap XML: {e}")
            # Fall back to text parsing if XML fails

    # Text output parsing
    # Function searches subject pattern
    # Method finds predicate matches
    # Regular expression locates object sections
    host_blocks = re.finditer(
        r"Nmap scan report for ([^\n]+).*?(?=Nmap scan report for|\Z)",
        output,
        re.DOTALL,
    )

    for block_match in host_blocks:
        block = block_match.group(0)
        host = block_match.group(1).strip()

        # Function extracts subject ports
        # Method finds predicate services
        # Regular expression locates object data
        port_matches = re.finditer(
            r"(\d+)/(\w+)\s+(\w+)\s+(\S+)(?:\s+(.+))?", block
        )

        for port_match in port_matches:
            # Function creates subject finding
            # Method builds predicate object
            # Dictionary stores object information
            finding = {
                "host": host,
                "port": int(port_match.group(1)),
                "protocol": port_match.group(2),
                "state": port_match.group(3),
                "service": port_match.group(4),
                "details": port_match.group(5) if port_match.group(5) else "",
                "timestamp": datetime.now().isoformat(),
                "tool": "nmap",
            }

            results.append(finding)

    return results


def _parse_nmap_xml(xml_output: str) -> List[Dict[str, Any]]:
    """
    Parse nmap XML output

    # Function parses subject XML
    # Method processes predicate format
    # Operation extracts object data

    Args:
        xml_output: nmap XML output

    Returns:
        List of parsed findings
    """
    results = []

    try:
        # Function parses subject XML
        # Method processes predicate data
        # Operation creates object tree
        root = ET.fromstring(xml_output)

        # Function finds subject hosts
        # Method locates predicate elements
        # Operation selects object nodes
        for host in root.findall(".//host"):
            addr = host.find(".//address")
            hostname_elem = host.find(".//hostname")

            # Function extracts subject address
            # Method retrieves predicate value
            # Operation gets object attribute
            ip_addr = addr.get("addr") if addr is not None else "unknown"
            hostname = (
                hostname_elem.get("name")
                if hostname_elem is not None
                else ip_addr
            )

            # Function processes subject ports
            # Method finds predicate elements
            # Operation iterates object list
            for port in host.findall(".//port"):
                state = port.find(".//state")
                service = port.find(".//service")

                if state is not None and state.get("state") == "open":
                    # Function creates subject result
                    # Method builds predicate object
                    # Dictionary stores object data
                    finding = {
                        "host": hostname,
                        "ip": ip_addr,
                        "port": int(port.get("portid", 0)),
                        "protocol": port.get("protocol", ""),
                        "state": state.get("state", ""),
                        "service": (
                            service.get("name", "")
                            if service is not None
                            else ""
                        ),
                        "product": (
                            service.get("product", "")
                            if service is not None
                            else ""
                        ),
                        "version": (
                            service.get("version", "")
                            if service is not None
                            else ""
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "tool": "nmap",
                    }

                    results.append(finding)

        # Function processes subject scripts
        # Method finds predicate output
        # Operation extracts object data
        for script in root.findall(".//script"):
            script_id = script.get("id", "")
            output = script.get("output", "")

            if "vuln" in script_id:
                # Function extracts subject host
                # Method finds predicate parent
                # Operation locates object element
                host_elem = script.find("./ancestor::host")
                if host_elem is not None:
                    addr = host_elem.find(".//address")
                    ip_addr = (
                        addr.get("addr") if addr is not None else "unknown"
                    )

                    # Function creates subject vulnerability
                    # Method builds predicate finding
                    # Dictionary stores object data
                    finding = {
                        "host": ip_addr,
                        "type": "vulnerability",
                        "title": script_id,
                        "description": output,
                        "severity": (
                            "High"
                            if "CRITICAL" in output or "HIGH" in output
                            else "Medium"
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "tool": "nmap",
                    }

                    results.append(finding)

    except Exception as e:
        logger.error(f"Error parsing nmap XML data: {e}")

    return results


def parse_dirb_results(output: str) -> List[Dict[str, Any]]:
    """
    Parse dirb web directory scanning results

    # Function parses subject results
    # Method extracts predicate data
    # Operation processes object text

    Args:
        output: dirb command output

    Returns:
        List of parsed findings
    """
    results = []

    # Function extracts subject target
    # Method finds predicate url
    # Regular expression locates object value
    target_match = re.search(r"TARGET: (https?://[^\s]+)", output)
    base_url = target_match.group(1) if target_match else "unknown"

    # Function finds subject discoveries
    # Method locates predicate matches
    # Regular expression extracts object data
    found_urls = re.finditer(r"\+ (.+?)\s+\(CODE:(\d+)\|SIZE:(\d+)\)", output)

    for match in found_urls:
        url_path = match.group(1)
        status_code = int(match.group(2))
        size = int(match.group(3))

        # Function creates subject finding
        # Method builds predicate object
        # Dictionary stores object information
        finding = {
            "url": (
                f"{base_url}/{url_path}"
                if not url_path.startswith("http")
                else url_path
            ),
            "path": url_path,
            "status_code": status_code,
            "size": size,
            "interesting": status_code in [200, 201, 301, 302, 401, 403],
            "timestamp": datetime.now().isoformat(),
            "tool": "dirb",
        }

        results.append(finding)

    return results


def parse_nikto_results(output: str) -> List[Dict[str, Any]]:
    """
    Parse nikto web vulnerability scanner results

    # Function parses subject results
    # Method extracts predicate data
    # Operation processes object text

    Args:
        output: nikto command output

    Returns:
        List of parsed findings
    """
    results = []

    # Function extracts subject target
    # Method finds predicate host
    # Regular expression locates object value
    target_match = re.search(r"Target: (https?://[^\s]+)", output)
    target = target_match.group(1) if target_match else "unknown"

    # Function finds subject items
    # Method locates predicate findings
    # Regular expression extracts object data
    findings = re.finditer(r"\+ (.+?): (.+)", output)

    for match in findings:
        code = match.group(1)
        description = match.group(2)

        # Function determines subject severity
        # Method analyzes predicate text
        # Operation assigns object rating
        severity = "Low"
        if any(
            kw in description.lower()
            for kw in ["xss", "sql injection", "command injection"]
        ):
            severity = "High"
        elif any(
            kw in description.lower()
            for kw in ["vulnerability", "exposed", "disclosure"]
        ):
            severity = "Medium"

        # Function creates subject finding
        # Method builds predicate object
        # Dictionary stores object information
        finding = {
            "target": target,
            "code": code,
            "description": description,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "tool": "nikto",
        }

        results.append(finding)

    return results


def parse_gobuster_results(output: str) -> List[Dict[str, Any]]:
    """
    Parse gobuster directory/file brute forcing results

    # Function parses subject results
    # Method extracts predicate data
    # Operation processes object text

    Args:
        output: gobuster command output

    Returns:
        List of parsed findings
    """
    results = []

    # Function extracts subject url
    # Method finds predicate target
    # Regular expression locates object value
    url_match = re.search(r"Url:\s+(.+)", output)
    base_url = url_match.group(1) if url_match else "unknown"

    # Function finds subject discoveries
    # Method locates predicate items
    # Regular expression extracts object data
    found_items = re.finditer(r"(https?://[^\s]+)\s+\(Status: (\d+)\)", output)

    for match in found_items:
        url = match.group(1)
        status = int(match.group(2))

        # Function creates subject finding
        # Method builds predicate object
        # Dictionary stores object information
        finding = {
            "url": url,
            "path": url.replace(base_url, ""),
            "status_code": status,
            "interesting": status in [200, 201, 301, 302, 401, 403],
            "timestamp": datetime.now().isoformat(),
            "tool": "gobuster",
        }

        results.append(finding)

    return results


def parse_whatweb_results(output: str) -> List[Dict[str, Any]]:
    """
    Parse whatweb website fingerprinting results

    # Function parses subject results
    # Method extracts predicate data
    # Operation processes object text

    Args:
        output: whatweb command output (text or JSON)

    Returns:
        List of parsed findings
    """
    results = []

    # Function checks subject format
    # Method detects predicate type
    # Condition tests object content
    if output.strip().startswith("{"):
        try:
            # Function parses subject JSON
            # Method processes predicate data
            # Operation deserializes object content
            json_data = json.loads(output)

            for url, data in json_data.items():
                # Function creates subject finding
                # Method builds predicate object
                # Dictionary stores object information
                finding = {
                    "url": url,
                    "technologies": [],
                    "timestamp": datetime.now().isoformat(),
                    "tool": "whatweb",
                }

                for plugin, info in data.get("plugins", {}).items():
                    if isinstance(info, dict) and "version" in info:
                        tech = f"{plugin} {info['version']}"
                    else:
                        tech = plugin

                    finding["technologies"].append(tech)

                results.append(finding)

            return results
        except Exception as e:
            logger.error(f"Error parsing WhatWeb JSON: {e}")
            # Fall back to text parsing

    # Function extracts subject sections
    # Method splits predicate output
    # Operation divides object text
    lines = output.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue

        # Function extracts subject url
        # Method finds predicate target
        # Regular expression locates object value
        url_match = re.match(r"http[s]?://[^\s]+", line)

        if not url_match:
            continue

        url = url_match.group(0)

        # Function extracts subject technologies
        # Method finds predicate items
        # Operation processes object text
        tech_text = line.replace(url, "").strip()
        technologies = []

        # Process brackets like [WordPress 5.4.2]
        for tech_match in re.finditer(r"\[([^\]]+)\]", tech_text):
            technologies.append(tech_match.group(1))

        # Function creates subject finding
        # Method builds predicate object
        # Dictionary stores object information
        finding = {
            "url": url,
            "technologies": technologies,
            "raw": tech_text,
            "timestamp": datetime.now().isoformat(),
            "tool": "whatweb",
        }

        results.append(finding)

    return results
