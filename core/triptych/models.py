"""
Triptych Models Module
--------------------
Defines the core models for the CTAS Triptych symbolic framework.
These models represent the UUID, CUID, and SCH identifiers that form
the basis of CTAS's symbolic cognition system.
"""

from typing import Dict, Any, List, Optional, Set, Union, Callable
from datetime import datetime, timedelta
import hashlib
import uuid
import json


class UUID:
    """
    UUID (The Identity Anchor) model.

    UUIDs provide permanent, immutable references for entities in the system.
    They serve as the foundational identity anchors for stateful reasoning.
    """

    def __init__(
        self,
        uuid_id: str,
        namespace: str,
        entity_type: str,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a UUID

        Args:
            uuid_id: The UUID identifier
            namespace: The namespace for this UUID
            entity_type: The type of entity this UUID represents
            ttl: Time-to-live for this UUID
            metadata: Additional metadata
        """
        self.uuid_id = uuid_id
        self.namespace = namespace
        self.entity_type = entity_type
        self.ttl = ttl
        self.metadata = metadata or {}
        self.creation_time = datetime.now()

    def is_expired(self) -> bool:
        """
        Check if this UUID has expired

        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False

        return datetime.now() > self.creation_time + self.ttl

    def get_expiration_time(self) -> Optional[datetime]:
        """
        Get the expiration time for this UUID

        Returns:
            Expiration time or None if no TTL
        """
        if self.ttl is None:
            return None

        return self.creation_time + self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation

        Returns:
            Dictionary representation
        """
        result = {
            "uuid_id": self.uuid_id,
            "namespace": self.namespace,
            "entity_type": self.entity_type,
            "creation_time": self.creation_time.isoformat(),
            "metadata": self.metadata,
        }

        if self.ttl is not None:
            result["ttl_seconds"] = self.ttl.total_seconds()
            result["expiration_time"] = self.get_expiration_time().isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UUID":
        """
        Create from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            UUID instance
        """
        ttl = None
        if "ttl_seconds" in data:
            ttl = timedelta(seconds=data["ttl_seconds"])

        instance = cls(
            uuid_id=data["uuid_id"],
            namespace=data["namespace"],
            entity_type=data["entity_type"],
            ttl=ttl,
            metadata=data.get("metadata", {}),
        )

        if "creation_time" in data:
            instance.creation_time = datetime.fromisoformat(
                data["creation_time"]
            )

        return instance

    def __str__(self) -> str:
        """String representation"""
        return self.uuid_id

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"UUID(id={self.uuid_id}, namespace={self.namespace}, type={self.entity_type})"

    def __eq__(self, other) -> bool:
        """Equality operator"""
        if not isinstance(other, UUID):
            return False
        return self.uuid_id == other.uuid_id

    def __hash__(self) -> int:
        """Hash function"""
        return hash(self.uuid_id)


class CUID:
    """
    CUID (The Contextual Fingerprint) model.

    CUIDs create semantic fingerprints for data-in-motion.
    They provide contextual continuity and serve as transient identifiers.
    """

    def __init__(
        self,
        cuid_id: str,
        source_data: Union[str, Dict[str, Any]],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a CUID

        Args:
            cuid_id: The CUID identifier
            source_data: The source data used to generate this CUID
            timestamp: Timestamp for this CUID
            metadata: Additional metadata
        """
        self.cuid_id = cuid_id
        self.source_data = source_data
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.deprecated = False
        self.successor_cuid = None

    def deprecate(self, successor_cuid: Optional[str] = None) -> None:
        """
        Deprecate this CUID

        Args:
            successor_cuid: Optional successor CUID ID
        """
        self.deprecated = True
        self.successor_cuid = successor_cuid

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation

        Returns:
            Dictionary representation
        """
        result = {
            "cuid_id": self.cuid_id,
            "timestamp": self.timestamp.isoformat(),
            "deprecated": self.deprecated,
            "metadata": self.metadata,
        }

        if isinstance(self.source_data, str):
            result["source_data_hash"] = hashlib.sha256(
                self.source_data.encode()
            ).hexdigest()
        else:
            result["source_data_hash"] = hashlib.sha256(
                json.dumps(self.source_data, sort_keys=True).encode()
            ).hexdigest()

        if self.successor_cuid:
            result["successor_cuid"] = self.successor_cuid

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CUID":
        """
        Create from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            CUID instance
        """
        # Note: source_data might not be available in the dictionary
        # as we only store the hash, not the actual data
        instance = cls(
            cuid_id=data["cuid_id"],
            source_data=data.get("source_data_hash", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

        instance.deprecated = data.get("deprecated", False)
        instance.successor_cuid = data.get("successor_cuid")

        return instance

    def __str__(self) -> str:
        """String representation"""
        return self.cuid_id

    def __repr__(self) -> str:
        """Detailed string representation"""
        status = "DEPRECATED" if self.deprecated else "ACTIVE"
        return f"CUID(id={self.cuid_id}, status={status})"

    def __eq__(self, other) -> bool:
        """Equality operator"""
        if not isinstance(other, CUID):
            return False
        return self.cuid_id == other.cuid_id

    def __hash__(self) -> int:
        """Hash function"""
        return hash(self.cuid_id)


class SCH:
    """
    SCH (The Synaptic Operator) model.

    SCHs represent symbolic cognition and operate as conditional operators.
    They enable cognitive activation and serve as the synaptic connections
    in the CTAS neural lattice.
    """

    def __init__(
        self,
        sch_id: str,
        domain: str,
        subdomain: str,
        entropy: float = 0.5,
        transition_readiness: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an SCH

        Args:
            sch_id: The SCH identifier
            domain: The domain for this SCH
            subdomain: The subdomain for this SCH
            entropy: Entropy (ζ) value (0-1)
            transition_readiness: Transition readiness (T) value (0-1)
            metadata: Additional metadata
        """
        self.sch_id = sch_id
        self.domain = domain
        self.subdomain = subdomain
        self.entropy = max(0.0, min(1.0, entropy))
        self.transition_readiness = max(0.0, min(1.0, transition_readiness))
        self.metadata = metadata or {}
        self.activation_record = []
        self.creation_time = datetime.now()

    def is_activated(self, tools_available: bool = True) -> bool:
        """
        Check if this SCH is activated based on the activation function
        Φh(ζ,T,tools) = 1 if ζ > 0.5 ∧ T > 0.7 ∧ tools available, 0 otherwise

        Args:
            tools_available: Whether required tools are available

        Returns:
            True if activated, False otherwise
        """
        activated = (
            self.entropy > 0.5
            and self.transition_readiness > 0.7
            and tools_available
        )

        # Record activation event
        self.activation_record.append(
            {
                "timestamp": datetime.now().isoformat(),
                "entropy": self.entropy,
                "transition_readiness": self.transition_readiness,
                "tools_available": tools_available,
                "activated": activated,
            }
        )

        return activated

    def update_entropy(self, value: float) -> None:
        """
        Update the entropy value

        Args:
            value: New entropy value (0-1)
        """
        self.entropy = max(0.0, min(1.0, value))

    def update_transition_readiness(self, value: float) -> None:
        """
        Update the transition readiness value

        Args:
            value: New transition readiness value (0-1)
        """
        self.transition_readiness = max(0.0, min(1.0, value))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation

        Returns:
            Dictionary representation
        """
        return {
            "sch_id": self.sch_id,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "entropy": self.entropy,
            "transition_readiness": self.transition_readiness,
            "metadata": self.metadata,
            "activation_record": self.activation_record,
            "creation_time": self.creation_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCH":
        """
        Create from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            SCH instance
        """
        instance = cls(
            sch_id=data["sch_id"],
            domain=data["domain"],
            subdomain=data["subdomain"],
            entropy=data["entropy"],
            transition_readiness=data["transition_readiness"],
            metadata=data.get("metadata", {}),
        )

        instance.activation_record = data.get("activation_record", [])

        if "creation_time" in data:
            instance.creation_time = datetime.fromisoformat(
                data["creation_time"]
            )

        return instance

    def __str__(self) -> str:
        """String representation"""
        return self.sch_id

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"SCH(id={self.sch_id}, domain={self.domain}.{self.subdomain}, ζ={self.entropy:.2f}, T={self.transition_readiness:.2f})"

    def __eq__(self, other) -> bool:
        """Equality operator"""
        if not isinstance(other, SCH):
            return False
        return self.sch_id == other.sch_id

    def __hash__(self) -> int:
        """Hash function"""
        return hash(self.sch_id)


class TriptychIdentity:
    """
    Combined Triptych identity (UUID + CUID + SCH).

    Represents a complete symbolic identity in the CTAS framework,
    with permanent anchor, contextual fingerprint, and cognitive activation.
    """

    def __init__(
        self, uuid_obj: UUID, cuid_obj: CUID, sch_obj: Optional[SCH] = None
    ):
        """
        Initialize a TriptychIdentity

        Args:
            uuid_obj: UUID instance
            cuid_obj: CUID instance
            sch_obj: Optional SCH instance
        """
        self.uuid = uuid_obj
        self.cuid = cuid_obj
        self.sch = sch_obj

    def is_activated(self, tools_available: bool = True) -> bool:
        """
        Check if this identity is activated

        Args:
            tools_available: Whether required tools are available

        Returns:
            True if activated, False otherwise
        """
        if self.sch is None:
            return False

        return self.sch.is_activated(tools_available)

    def is_expired(self) -> bool:
        """
        Check if this identity has expired

        Returns:
            True if expired, False otherwise
        """
        return self.uuid.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation

        Returns:
            Dictionary representation
        """
        result = {"uuid": self.uuid.to_dict(), "cuid": self.cuid.to_dict()}

        if self.sch is not None:
            result["sch"] = self.sch.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriptychIdentity":
        """
        Create from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            TriptychIdentity instance
        """
        uuid_obj = UUID.from_dict(data["uuid"])
        cuid_obj = CUID.from_dict(data["cuid"])
        sch_obj = None

        if "sch" in data:
            sch_obj = SCH.from_dict(data["sch"])

        return cls(uuid_obj, cuid_obj, sch_obj)

    def __str__(self) -> str:
        """String representation"""
        parts = [f"UUID:{self.uuid}", f"CUID:{self.cuid}"]
        if self.sch:
            parts.append(f"SCH:{self.sch}")
        return f"Triptych({', '.join(parts)})"
