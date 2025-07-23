"""
NyxTrace Collector Interfaces
--------------------------
Interfaces for data collection components.
"""

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from core.interfaces.base import NyxTraceComponent, Registrable


@dataclass
class EEI:
    """
    Essential Element of Information (EEI).

    This class represents a piece of intelligence data
    collected from a source.
    """

    eei_id: str
    source_id: str
    collection_time: datetime.datetime
    data: Any
    confidence: float = 1.0
    priority: str = "Medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate instance after initialization"""
        if not self.eei_id:
            self.eei_id = f"eei-{uuid.uuid4()}"

        if not self.source_id:
            self.source_id = "unknown"

        if not self.collection_time:
            self.collection_time = datetime.datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "eei_id": self.eei_id,
            "source_id": self.source_id,
            "collection_time": self.collection_time.isoformat(),
            "data": self.data,
            "confidence": self.confidence,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EEI":
        """Create from dictionary representation"""
        if "collection_time" in data and isinstance(
            data["collection_time"], str
        ):
            data["collection_time"] = datetime.datetime.fromisoformat(
                data["collection_time"]
            )

        return cls(**data)


@dataclass
class CollectorParams:
    """
    Parameters for data collection operations.

    This class defines the parameters for a collection
    operation, including targets, time constraints, and
    custom parameters.
    """

    targets: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None
    priority: str = "Medium"
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate instance after initialization"""
        # Validate priority
        valid_priorities = ["Critical", "High", "Medium", "Low"]
        if self.priority not in valid_priorities:
            self.priority = "Medium"


class Collector(NyxTraceComponent, Registrable):
    """
    Interface for data collectors.

    Data collectors are responsible for retrieving data from
    external sources and converting it into EEIs.
    """

    async def collect(self, params: CollectorParams) -> List[EEI]:
        """
        Collect data based on parameters

        Args:
            params: Collection parameters

        Returns:
            List of collected EEIs
        """
        raise NotImplementedError()

    def get_collection_domain(self) -> str:
        """Get the domain this collector operates in"""
        raise NotImplementedError()

    def get_collection_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this collector"""
        raise NotImplementedError()

    def validate_parameters(
        self, params: CollectorParams
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate collection parameters

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError()
