"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-CYBERWARFARE-REPORTING-0001    │
// │ 📁 domain       : Cyberwarfare, Offensive Security, Analysis│
// │ 🧠 description  : Cyberwarfare assessment reporting module  │
// │                  Handles creation and export of reports     │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CYBERWARFARE                        │
// │ 🧩 dependencies : pandas, jinja2, datetime                 │
// │ 🔧 tool_usage   : Analysis, Reporting                      │
// │ 📡 input_type   : Assessment data, scan results            │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : analysis, reporting, summarization       │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Cyberwarfare Assessment Reporting Module
----------------------------------------
This module provides functionality for creating, formatting, and exporting
security assessment reports based on vulnerability scanning and penetration
testing results. It supports multiple output formats and template-based
report generation.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import logging

# Setup module logger
logger = logging.getLogger(__name__)


class CyberReport:
    """
    Class to manage a comprehensive security assessment report

    # Class manages subject report
    # Structure organizes predicate data
    # Object contains vulnerability information
    """

    def __init__(
        self,
        title: str,
        target: str,
        assessment_date: Optional[datetime] = None,
    ):
        """
        Initialize a new security assessment report

        # Function initializes subject report
        # Method creates predicate object
        # Operation prepares object state

        Args:
            title: Title of the report
            target: Target system or network
            assessment_date: Date of assessment (defaults to now)
        """
        self.title = title
        self.target = target
        self.assessment_date = assessment_date or datetime.now()
        self.findings: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "generated_by": "CTAS NyxTrace Cyberwarfare Module",
            "creation_date": datetime.now(),
            "ctas_version": "6.5",
            "classification": "RESTRICTED",
        }

    def add_finding(
        self,
        title: str,
        severity: str,
        description: str,
        cvss_score: Optional[float] = None,
        remediation: Optional[str] = None,
        affected_components: Optional[List[str]] = None,
    ) -> None:
        """
        Add a security finding to the report

        # Function adds subject finding
        # Method appends predicate vulnerability
        # Operation extends object list

        Args:
            title: Title of the vulnerability
            severity: Severity rating (Critical, High, Medium, Low, Info)
            description: Description of the vulnerability
            cvss_score: CVSS score if available
            remediation: Remediation steps
            affected_components: List of affected components
        """
        # Function creates subject finding
        # Method builds predicate data
        # Dictionary structures object information
        finding = {
            "id": f"FINDING-{len(self.findings) + 1:04d}",
            "title": title,
            "severity": severity,
            "description": description,
            "cvss_score": cvss_score,
            "remediation": remediation or "No remediation steps provided.",
            "affected_components": affected_components or [],
            "date_added": datetime.now(),
        }

        # Function adds subject finding
        # Method appends predicate data
        # Operation updates object state
        self.findings.append(finding)

        # Function logs subject addition
        # Method records predicate action
        # Message documents object change
        logger.info(f"Added finding '{title}' with severity {severity}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary

        # Function converts subject report
        # Method transforms predicate data
        # Operation serializes object state

        Returns:
            Dictionary representation of the report
        """
        return {
            "title": self.title,
            "target": self.target,
            "assessment_date": self.assessment_date.isoformat(),
            "metadata": self.metadata,
            "findings": self.findings,
            "statistics": {
                "total_findings": len(self.findings),
                "by_severity": self._count_by_severity(),
            },
        }

    def _count_by_severity(self) -> Dict[str, int]:
        """
        Count findings by severity

        # Function counts subject findings
        # Method groups predicate data
        # Operation aggregates object values

        Returns:
            Dictionary with counts by severity
        """
        result = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}

        for finding in self.findings:
            if finding["severity"] in result:
                result[finding["severity"]] += 1

        return result


def create_vulnerability_report(
    scan_results: Dict[str, Any], target: str
) -> CyberReport:
    """
    Create a vulnerability report from scan results

    # Function creates subject report
    # Method processes predicate results
    # Operation generates object document

    Args:
        scan_results: Results from vulnerability scanning tools
        target: Target system or network

    Returns:
        CyberReport object with findings
    """
    # Function creates subject report
    # Method initializes predicate object
    # Operation prepares object container
    report = CyberReport(
        title=f"Vulnerability Assessment: {target}", target=target
    )

    # Function checks subject input
    # Method validates predicate data
    # Condition tests object presence
    if not scan_results:
        logger.warning("No scan results provided for report generation")
        return report

    # Function processes subject results
    # Method extracts predicate findings
    # Operation transforms object data
    for tool_name, tool_results in scan_results.items():
        if not tool_results:
            continue

        for item in tool_results:
            # Function sets subject severity
            # Method computes predicate rating
            # Operation determines object level
            severity = "Medium"  # Default severity
            if "severity" in item:
                severity = item["severity"]
            elif "risk" in item:
                severity = item["risk"]
            elif "cvss" in item and item["cvss"] >= 7.0:
                severity = "High"
            elif "cvss" in item and item["cvss"] >= 4.0:
                severity = "Medium"
            elif "cvss" in item:
                severity = "Low"

            # Function adds subject finding
            # Method records predicate vulnerability
            # Operation updates object report
            report.add_finding(
                title=item.get(
                    "title", item.get("name", f"Finding from {tool_name}")
                ),
                severity=severity,
                description=item.get("description", "No description provided"),
                cvss_score=item.get("cvss"),
                remediation=item.get("remediation"),
                affected_components=item.get("affected", []),
            )

    return report


def export_report(
    report: CyberReport,
    format_type: str = "json",
    output_path: Optional[str] = None,
) -> str:
    """
    Export a report in the specified format

    # Function exports subject report
    # Method formats predicate data
    # Operation saves object file

    Args:
        report: CyberReport object to export
        format_type: Output format (json, csv, html)
        output_path: Path to save the report (optional)

    Returns:
        Path to the exported file or the report content as string
    """
    # Function prepares subject data
    # Method serializes predicate report
    # Operation converts object values
    report_data = report.to_dict()

    # Function generates subject filename
    # Method creates predicate path
    # Operation formats object string
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = (
            report.target.replace(".", "_").replace("/", "_").replace(":", "_")
        )
        output_path = (
            f"reports/vulnerability_{target_name}_{timestamp}.{format_type}"
        )

        # Function ensures subject directory
        # Method creates predicate folder
        # Operation prepares object location
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Function exports subject format
    # Method handles predicate type
    # Condition selects object action
    if format_type.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
    elif format_type.lower() == "csv":
        # Convert findings to DataFrame and export
        findings_df = pd.DataFrame(report_data["findings"])
        findings_df.to_csv(output_path, index=False)
    elif format_type.lower() == "html":
        # Basic HTML template (in a real implementation, use Jinja2)
        html_content = f"""
        <html>
        <head><title>{report_data['title']}</title></head>
        <body>
        <h1>{report_data['title']}</h1>
        <p>Target: {report_data['target']}</p>
        <p>Date: {report_data['assessment_date']}</p>
        <h2>Findings</h2>
        <ul>
        {"".join([f"<li><b>{f['title']}</b> ({f['severity']}): {f['description']}</li>" for f in report_data['findings']])}
        </ul>
        </body>
        </html>
        """
        with open(output_path, "w") as f:
            f.write(html_content)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")

    # Function logs subject export
    # Method records predicate action
    # Message documents object action
    logger.info(f"Exported report to {output_path} in {format_type} format")

    return output_path


def generate_report_summary(report: CyberReport) -> Dict[str, Any]:
    """
    Generate a summary of the report

    # Function generates subject summary
    # Method extracts predicate highlights
    # Operation creates object overview

    Args:
        report: CyberReport object to summarize

    Returns:
        Dictionary with report summary
    """
    # Function prepares subject data
    # Method extracts predicate metrics
    # Operation computes object statistics
    severity_counts = report._count_by_severity()
    high_risk_count = severity_counts["Critical"] + severity_counts["High"]

    # Function calculates subject risk
    # Method determines predicate level
    # Operation assesses object score
    if high_risk_count > 5:
        risk_level = "Critical"
    elif high_risk_count > 0:
        risk_level = "High"
    elif severity_counts["Medium"] > 3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Function creates subject summary
    # Method formats predicate data
    # Dictionary structures object information
    return {
        "title": report.title,
        "target": report.target,
        "assessment_date": report.assessment_date.isoformat(),
        "finding_count": len(report.findings),
        "severity_counts": severity_counts,
        "overall_risk_level": risk_level,
        "high_priority_findings": [
            f for f in report.findings if f["severity"] in ("Critical", "High")
        ][:5],
    }
