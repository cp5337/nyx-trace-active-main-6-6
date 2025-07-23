"""
Analyzer Interfaces Module
-------------------------
Defines the interfaces for intelligence analysis components in the NyxTrace system.
These analyzers evaluate processed intelligence to identify patterns, threats, and insights.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Tuple
from datetime import datetime

from core.interfaces.base import (
    NyxTraceComponent,
    Registrable,
    LifecycleManaged,
    Configurable,
    EventEmitter,
    CTASIntegrated,
)
from core.interfaces.processors import ProcessedIntelligence

T = TypeVar("T")


class AnalysisResult:
    """
    Intelligence analysis result data structure.
    Contains analysis findings, insights, and recommendations.
    """

    def __init__(
        self,
        result_id: str,
        analyzer_id: str,
        analysis_time: datetime,
        input_data: List[ProcessedIntelligence],
        findings: Dict[str, Any],
        insights: List[str],
        recommendations: List[str],
        confidence: float = 0.5,
        priority: str = "Medium",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an AnalysisResult instance

        Args:
            result_id: Unique identifier for this analysis result
            analyzer_id: Identifier of the analyzer that produced this result
            analysis_time: Time of analysis
            input_data: Input data used for analysis
            findings: Structured findings from analysis
            insights: List of insights derived from findings
            recommendations: List of recommended actions
            confidence: Confidence score (0-1)
            priority: Priority level ("High", "Medium", "Low")
            tags: Tags for categorization
            metadata: Additional metadata
        """
        self.result_id = result_id
        self.analyzer_id = analyzer_id
        self.analysis_time = analysis_time
        self.input_data = input_data
        self.findings = findings
        self.insights = insights
        self.recommendations = recommendations
        self.confidence = confidence
        self.priority = priority
        self.tags = tags or []
        self.metadata = metadata or {}

        # CTAS integration fields
        self.entropy = 0.5
        self.transition_readiness = 0.5
        self.uuid = None
        self.cuid = None
        self.sch = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "result_id": self.result_id,
            "analyzer_id": self.analyzer_id,
            "analysis_time": self.analysis_time.isoformat(),
            "input_data": (
                [intel.to_dict() for intel in self.input_data]
                if self.input_data
                else []
            ),
            "findings": self.findings,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
            "entropy": self.entropy,
            "transition_readiness": self.transition_readiness,
            "uuid": self.uuid,
            "cuid": self.cuid,
            "sch": self.sch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary representation"""
        # Convert input_data dicts to ProcessedIntelligence objects
        input_data = [
            ProcessedIntelligence.from_dict(intel)
            for intel in data.get("input_data", [])
        ]

        instance = cls(
            result_id=data["result_id"],
            analyzer_id=data["analyzer_id"],
            analysis_time=datetime.fromisoformat(data["analysis_time"]),
            input_data=input_data,
            findings=data["findings"],
            insights=data["insights"],
            recommendations=data["recommendations"],
            confidence=data["confidence"],
            priority=data["priority"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

        # Set CTAS integration fields
        instance.entropy = data.get("entropy", 0.5)
        instance.transition_readiness = data.get("transition_readiness", 0.5)
        instance.uuid = data.get("uuid")
        instance.cuid = data.get("cuid")
        instance.sch = data.get("sch")

        return instance


class AnalyzerParams:
    """Parameters for an analysis operation"""

    def __init__(
        self,
        analysis_depth: str = "standard",
        confidence_threshold: float = 0.0,
        priority_threshold: str = "Low",
        time_window: Optional[Tuple[datetime, datetime]] = None,
        include_recommendations: bool = True,
        max_insights: int = 10,
        custom_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize analyzer parameters

        Args:
            analysis_depth: Depth of analysis ("minimal", "standard", "deep")
            confidence_threshold: Minimum confidence threshold for results
            priority_threshold: Minimum priority threshold ("Low", "Medium", "High")
            time_window: Optional time window for analysis
            include_recommendations: Whether to include recommendations
            max_insights: Maximum number of insights to generate
            custom_params: Additional custom parameters
        """
        self.analysis_depth = analysis_depth
        self.confidence_threshold = confidence_threshold
        self.priority_threshold = priority_threshold
        self.time_window = time_window
        self.include_recommendations = include_recommendations
        self.max_insights = max_insights
        self.custom_params = custom_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "analysis_depth": self.analysis_depth,
            "confidence_threshold": self.confidence_threshold,
            "priority_threshold": self.priority_threshold,
            "include_recommendations": self.include_recommendations,
            "max_insights": self.max_insights,
            "custom_params": self.custom_params,
        }

        if self.time_window:
            result["time_window_start"] = self.time_window[0].isoformat()
            result["time_window_end"] = self.time_window[1].isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalyzerParams":
        """Create from dictionary representation"""
        # Handle time window if present
        time_window = None
        if "time_window_start" in data and "time_window_end" in data:
            start = datetime.fromisoformat(data["time_window_start"])
            end = datetime.fromisoformat(data["time_window_end"])
            time_window = (start, end)

        return cls(
            analysis_depth=data.get("analysis_depth", "standard"),
            confidence_threshold=data.get("confidence_threshold", 0.0),
            priority_threshold=data.get("priority_threshold", "Low"),
            time_window=time_window,
            include_recommendations=data.get("include_recommendations", True),
            max_insights=data.get("max_insights", 10),
            custom_params=data.get("custom_params"),
        )


class Analyzer(
    NyxTraceComponent,
    Registrable,
    LifecycleManaged,
    Configurable,
    EventEmitter,
    CTASIntegrated,
):
    """
    Base interface for all intelligence analyzers in NyxTrace.

    Analyzers evaluate processed intelligence to identify patterns,
    threats, and insights that can inform decision-making.
    """

    @abstractmethod
    async def analyze(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> AnalysisResult:
        """
        Analyze processed intelligence

        Args:
            intelligence: List of processed intelligence to analyze
            params: Analysis parameters

        Returns:
            Analysis result
        """
        pass

    @abstractmethod
    def get_analysis_domain(self) -> str:
        """
        Get the domain this analyzer operates in

        Returns:
            Domain string (e.g., "threat", "pattern", "anomaly", etc.)
        """
        pass

    @abstractmethod
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this analyzer

        Returns:
            Dictionary of capabilities
        """
        pass

    @abstractmethod
    def validate_parameters(
        self, params: AnalyzerParams
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate analysis parameters

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class ThreatAnalyzer(Analyzer):
    """
    Interface for analyzers focused on threat intelligence analysis.
    """

    @abstractmethod
    async def analyze_threat_indicators(
        self,
        indicators: List[Dict[str, Any]],
        params: Optional[AnalyzerParams] = None,
    ) -> Dict[str, Any]:
        """
        Analyze threat indicators

        Args:
            indicators: List of threat indicators
            params: Analysis parameters

        Returns:
            Threat analysis results
        """
        pass

    @abstractmethod
    async def assess_threat_level(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> Dict[str, Any]:
        """
        Assess threat level

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            Threat level assessment
        """
        pass

    @abstractmethod
    async def identify_threat_actors(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify potential threat actors

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            List of identified threat actors
        """
        pass

    @abstractmethod
    async def predict_attack_vector(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict potential attack vectors

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            List of predicted attack vectors
        """
        pass


class PatternAnalyzer(Analyzer):
    """
    Interface for analyzers focused on pattern detection and analysis.
    """

    @abstractmethod
    async def detect_temporal_patterns(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect temporal patterns

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            List of detected temporal patterns
        """
        pass

    @abstractmethod
    async def detect_spatial_patterns(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect spatial patterns

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            List of detected spatial patterns
        """
        pass

    @abstractmethod
    async def detect_behavioral_patterns(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect behavioral patterns

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            List of detected behavioral patterns
        """
        pass

    @abstractmethod
    async def correlate_patterns(
        self,
        patterns: List[Dict[str, Any]],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Correlate detected patterns

        Args:
            patterns: List of patterns to correlate
            params: Analysis parameters

        Returns:
            List of correlated patterns
        """
        pass


class AnomalyAnalyzer(Analyzer):
    """
    Interface for analyzers focused on anomaly detection and analysis.
    """

    @abstractmethod
    async def detect_anomalies(
        self,
        intelligence: List[ProcessedIntelligence],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies

        Args:
            intelligence: List of processed intelligence
            params: Analysis parameters

        Returns:
            List of detected anomalies
        """
        pass

    @abstractmethod
    async def classify_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify detected anomalies

        Args:
            anomalies: List of anomalies to classify
            params: Analysis parameters

        Returns:
            List of classified anomalies
        """
        pass

    @abstractmethod
    async def assess_anomaly_impact(
        self,
        anomalies: List[Dict[str, Any]],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assess the impact of detected anomalies

        Args:
            anomalies: List of anomalies to assess
            params: Analysis parameters

        Returns:
            List of impact assessments
        """
        pass

    @abstractmethod
    async def generate_anomaly_alerts(
        self,
        anomalies: List[Dict[str, Any]],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for detected anomalies

        Args:
            anomalies: List of anomalies to generate alerts for
            params: Analysis parameters

        Returns:
            List of generated alerts
        """
        pass


class PredictiveAnalyzer(Analyzer):
    """
    Interface for analyzers focused on predictive analysis.
    """

    @abstractmethod
    async def predict_future_activities(
        self,
        intelligence: List[ProcessedIntelligence],
        time_horizon: str,
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict future activities

        Args:
            intelligence: List of processed intelligence
            time_horizon: Time horizon for predictions (e.g., "short", "medium", "long")
            params: Analysis parameters

        Returns:
            List of predicted activities
        """
        pass

    @abstractmethod
    async def forecast_trends(
        self,
        intelligence: List[ProcessedIntelligence],
        trend_types: List[str],
        params: Optional[AnalyzerParams] = None,
    ) -> List[Dict[str, Any]]:
        """
        Forecast trends

        Args:
            intelligence: List of processed intelligence
            trend_types: Types of trends to forecast
            params: Analysis parameters

        Returns:
            List of forecasted trends
        """
        pass

    @abstractmethod
    async def assess_risk(
        self,
        intelligence: List[ProcessedIntelligence],
        risk_factors: List[str],
        params: Optional[AnalyzerParams] = None,
    ) -> Dict[str, Any]:
        """
        Assess risks

        Args:
            intelligence: List of processed intelligence
            risk_factors: Risk factors to consider
            params: Analysis parameters

        Returns:
            Risk assessment
        """
        pass
