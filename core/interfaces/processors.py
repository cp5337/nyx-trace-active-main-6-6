"""
NyxTrace Processor Interfaces
--------------------------
Interfaces for data processing components.
"""

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from core.interfaces.base import NyxTraceComponent, Registrable
from core.interfaces.collectors import EEI


@dataclass
class ProcessedIntelligence:
    """
    Processed intelligence data.

    This class represents intelligence data that has been
    processed from raw EEIs.
    """

    intelligence_id: str
    processor_id: str
    processing_time: datetime.datetime
    raw_data: EEI
    processed_data: Dict[str, Any]
    confidence: float = 1.0
    priority: str = "Medium"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate instance after initialization"""
        if not self.intelligence_id:
            self.intelligence_id = f"intel-{uuid.uuid4()}"

        if not self.processor_id:
            self.processor_id = "unknown"

        if not self.processing_time:
            self.processing_time = datetime.datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "intelligence_id": self.intelligence_id,
            "processor_id": self.processor_id,
            "processing_time": self.processing_time.isoformat(),
            "raw_data": self.raw_data.to_dict() if self.raw_data else None,
            "processed_data": self.processed_data,
            "confidence": self.confidence,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
            "analysis_results": self.analysis_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedIntelligence":
        """Create from dictionary representation"""
        if "processing_time" in data and isinstance(
            data["processing_time"], str
        ):
            data["processing_time"] = datetime.datetime.fromisoformat(
                data["processing_time"]
            )

        if "raw_data" in data and isinstance(data["raw_data"], dict):
            data["raw_data"] = EEI.from_dict(data["raw_data"])

        return cls(**data)

    def add_analysis_result(self, key: str, result: Any) -> None:
        """
        Add an analysis result

        Args:
            key: Result key
            result: Analysis result
        """
        self.analysis_results[key] = result

    def get_analysis_result(self, key: str, default: Any = None) -> Any:
        """
        Get an analysis result

        Args:
            key: Result key
            default: Default value if result not found

        Returns:
            Analysis result
        """
        return self.analysis_results.get(key, default)


@dataclass
class ProcessorParams:
    """
    Parameters for data processing operations.

    This class defines the parameters for a processing
    operation, including options and custom parameters.
    """

    options: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


class Processor(NyxTraceComponent, Registrable):
    """
    Interface for data processors.

    Data processors are responsible for transforming raw EEIs
    into structured intelligence.
    """

    async def process(
        self, eeis: List[EEI], params: Optional[ProcessorParams] = None
    ) -> List[ProcessedIntelligence]:
        """
        Process a list of EEIs into structured intelligence

        Args:
            eeis: List of EEIs to process
            params: Processing parameters

        Returns:
            List of processed intelligence
        """
        raise NotImplementedError()

    async def process_single(
        self, eei: EEI, params: Optional[ProcessorParams] = None
    ) -> ProcessedIntelligence:
        """
        Process a single EEI into structured intelligence

        Args:
            eei: EEI to process
            params: Processing parameters

        Returns:
            Processed intelligence
        """
        raise NotImplementedError()

    def get_processing_domain(self) -> str:
        """Get the domain this processor operates in"""
        raise NotImplementedError()

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this processor"""
        raise NotImplementedError()

    def validate_parameters(
        self, params: ProcessorParams
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate processing parameters

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError()
