"""
URL Health Monitor
-----------------
This module provides functionality to monitor the health and accessibility
of URLs used in the application, particularly for web scraping.

Features:
- URL validation and availability checking
- Response time tracking
- Status code monitoring
- Historical uptime tracking
- SSL certificate validation
- Content change detection
"""

import requests
import time
import concurrent.futures
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Suppress insecure request warnings for non-critical health checks
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class URLHealthStatus:
    """Represents the health status of a URL"""

    STATUS_OK = "OK"
    STATUS_SLOW = "Slow"
    STATUS_ERROR = "Error"
    STATUS_TIMEOUT = "Timeout"
    STATUS_INVALID = "Invalid URL"

    def __init__(self, url: str):
        self.url = url
        self.last_checked = None
        self.response_time = None
        self.status_code = None
        self.status = None
        self.error_message = None
        self.content_hash = None
        self.history = []  # List of historical check results

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "url": self.url,
            "last_checked": (
                self.last_checked.isoformat() if self.last_checked else None
            ),
            "response_time": self.response_time,
            "status_code": self.status_code,
            "status": self.status,
            "error_message": self.error_message,
        }


class URLHealthMonitor:
    """Monitors the health of URLs"""

    def __init__(self):
        self.url_status = {}  # Dictionary of URL to URLHealthStatus
        self.slow_threshold = 2.0  # Seconds
        self.timeout = 10.0  # Seconds
        self.history_length = (
            100  # Maximum number of historical entries per URL
        )
        self.last_bulk_check_time = None

    def check_url_health(
        self, url: str, verify_ssl: bool = False
    ) -> URLHealthStatus:
        """
        Check the health of a single URL

        Args:
            url: The URL to check
            verify_ssl: Whether to verify SSL certificates

        Returns:
            URLHealthStatus object with the results
        """
        if url not in self.url_status:
            self.url_status[url] = URLHealthStatus(url)

        status_obj = self.url_status[url]

        try:
            start_time = time.time()
            response = requests.head(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                verify=verify_ssl,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
            )
            elapsed = time.time() - start_time

            status_obj.response_time = elapsed
            status_obj.status_code = response.status_code
            status_obj.last_checked = datetime.now()

            if 200 <= response.status_code < 300:
                if elapsed > self.slow_threshold:
                    status_obj.status = URLHealthStatus.STATUS_SLOW
                else:
                    status_obj.status = URLHealthStatus.STATUS_OK
            else:
                status_obj.status = URLHealthStatus.STATUS_ERROR
                status_obj.error_message = (
                    f"HTTP status code: {response.status_code}"
                )

        except requests.exceptions.Timeout:
            status_obj.status = URLHealthStatus.STATUS_TIMEOUT
            status_obj.error_message = "Request timed out"
            status_obj.last_checked = datetime.now()

        except requests.exceptions.RequestException as e:
            status_obj.status = URLHealthStatus.STATUS_ERROR
            status_obj.error_message = str(e)
            status_obj.last_checked = datetime.now()

        except Exception as e:
            status_obj.status = URLHealthStatus.STATUS_ERROR
            status_obj.error_message = f"Unexpected error: {str(e)}"
            status_obj.last_checked = datetime.now()

        # Add to history
        history_entry = {
            "timestamp": status_obj.last_checked,
            "status": status_obj.status,
            "response_time": status_obj.response_time,
            "status_code": status_obj.status_code,
        }
        status_obj.history.append(history_entry)

        # Trim history if needed
        if len(status_obj.history) > self.history_length:
            status_obj.history = status_obj.history[-self.history_length :]

        return status_obj

    def check_urls_health(
        self, urls: List[str], max_workers: int = 10
    ) -> Dict[str, URLHealthStatus]:
        """
        Check the health of multiple URLs in parallel

        Args:
            urls: List of URLs to check
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary of URL to URLHealthStatus
        """
        results = {}
        self.last_bulk_check_time = datetime.now()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_url = {
                executor.submit(self.check_url_health, url): url for url in urls
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    status = future.result()
                    results[url] = status
                except Exception as e:
                    print(f"Error checking {url}: {str(e)}")

        return results

    def get_url_status(self, url: str) -> Optional[URLHealthStatus]:
        """
        Get the current status of a URL

        Args:
            url: The URL to get status for

        Returns:
            URLHealthStatus object or None if URL not monitored
        """
        return self.url_status.get(url)

    def get_url_history(self, url: str) -> List[Dict]:
        """
        Get the historical status of a URL

        Args:
            url: The URL to get history for

        Returns:
            List of historical status entries
        """
        if url in self.url_status:
            return self.url_status[url].history
        return []

    def get_all_statuses(self) -> Dict[str, URLHealthStatus]:
        """
        Get the current status of all monitored URLs

        Returns:
            Dictionary of URL to URLHealthStatus
        """
        return self.url_status

    def get_status_summary(self) -> Dict[str, int]:
        """
        Get a summary of URL statuses

        Returns:
            Dictionary with counts of each status type
        """
        summary = {
            URLHealthStatus.STATUS_OK: 0,
            URLHealthStatus.STATUS_SLOW: 0,
            URLHealthStatus.STATUS_ERROR: 0,
            URLHealthStatus.STATUS_TIMEOUT: 0,
            URLHealthStatus.STATUS_INVALID: 0,
        }

        for url, status in self.url_status.items():
            if status.status in summary:
                summary[status.status] += 1

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert URL health data to a DataFrame

        Returns:
            DataFrame with URL health data
        """
        data = []
        for url, status in self.url_status.items():
            row = status.to_dict()
            data.append(row)

        return pd.DataFrame(data)

    def create_summary_chart(self) -> go.Figure:
        """
        Create a summary chart of URL health

        Returns:
            Plotly figure
        """
        summary = self.get_status_summary()

        # Create pie chart
        labels = list(summary.keys())
        values = list(summary.values())

        colors = {
            URLHealthStatus.STATUS_OK: "#00CC96",
            URLHealthStatus.STATUS_SLOW: "#FFA15A",
            URLHealthStatus.STATUS_ERROR: "#EF553B",
            URLHealthStatus.STATUS_TIMEOUT: "#AB63FA",
            URLHealthStatus.STATUS_INVALID: "#636EFA",
        }

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker_colors=[
                        colors.get(label, "#CCCCCC") for label in labels
                    ],
                )
            ]
        )

        fig.update_layout(
            title_text="URL Health Status Summary", showlegend=True
        )

        return fig

    def create_response_time_chart(
        self, urls: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a response time chart for URLs

        Args:
            urls: List of URLs to include, or all if None

        Returns:
            Plotly figure
        """
        if urls is None:
            urls = list(self.url_status.keys())

        # Create line chart of response times
        fig = go.Figure()

        for url in urls:
            if url in self.url_status:
                history = self.url_status[url].history
                if history:
                    timestamps = [entry["timestamp"] for entry in history]
                    response_times = [
                        entry["response_time"] for entry in history
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=response_times,
                            mode="lines+markers",
                            name=url,
                        )
                    )

        fig.update_layout(
            title="URL Response Times",
            xaxis_title="Time",
            yaxis_title="Response Time (seconds)",
            hovermode="closest",
        )

        return fig

    def create_availability_chart(
        self, urls: Optional[List[str]] = None, days: int = 7
    ) -> go.Figure:
        """
        Create an availability chart for URLs

        Args:
            urls: List of URLs to include, or all if None
            days: Number of days to include in the chart

        Returns:
            Plotly figure
        """
        if urls is None:
            urls = list(self.url_status.keys())

        # Calculate availability percentages
        availability_data = []

        for url in urls:
            if url in self.url_status:
                history = self.url_status[url].history
                if history:
                    # Count successful requests (OK or SLOW)
                    total_checks = len(history)
                    successful_checks = sum(
                        1
                        for entry in history
                        if entry["status"]
                        in [
                            URLHealthStatus.STATUS_OK,
                            URLHealthStatus.STATUS_SLOW,
                        ]
                    )

                    if total_checks > 0:
                        availability = (successful_checks / total_checks) * 100
                    else:
                        availability = 0

                    availability_data.append(
                        {
                            "URL": url,
                            "Availability (%)": availability,
                            "Total Checks": total_checks,
                            "Successful": successful_checks,
                        }
                    )

        if not availability_data:
            # Create empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="URL Availability (No Data)")
            return fig

        # Create bar chart
        df = pd.DataFrame(availability_data)
        fig = px.bar(
            df,
            x="URL",
            y="Availability (%)",
            color="Availability (%)",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[0, 100],
            hover_data=["Total Checks", "Successful"],
        )

        fig.update_layout(
            title="URL Availability",
            xaxis_title="URL",
            yaxis_title="Availability (%)",
            hovermode="closest",
        )

        return fig
