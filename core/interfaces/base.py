"""
NyxTrace Base Interfaces
----------------------
Base interfaces for the NyxTrace platform components.
"""

import uuid
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)


@runtime_checkable
class NyxTraceComponent(Protocol):
    """
    Base interface for all NyxTrace components.

    All components in the NyxTrace system must implement this interface
    to provide basic identification and metadata.
    """

    def get_id(self) -> str:
        """Get the component's unique identifier"""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Get the component's metadata"""
        ...

    def get_activation_parameters(self) -> Dict[str, float]:
        """Get the component's activation parameters"""
        ...

    def is_activated(self) -> bool:
        """Check if the component is currently activated"""
        ...


@runtime_checkable
class Registrable(Protocol):
    """
    Interface for components that can register with a registry.
    """

    def register(self, registry: Any) -> str:
        """
        Register with a registry

        Args:
            registry: Registry to register with

        Returns:
            Component ID
        """
        ...

    def unregister(self, registry: Any) -> bool:
        """
        Unregister from a registry

        Args:
            registry: Registry to unregister from

        Returns:
            Success flag
        """
        ...


@runtime_checkable
class LifecycleManaged(Protocol):
    """
    Interface for components with managed lifecycles.
    """

    def initialize(self) -> bool:
        """
        Initialize the component

        Returns:
            Success flag
        """
        ...

    def shutdown(self) -> bool:
        """
        Shutdown the component

        Returns:
            Success flag
        """
        ...


@runtime_checkable
class Configurable(Protocol):
    """
    Interface for components that can be configured.
    """

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the component

        Args:
            config: Configuration parameters

        Returns:
            Success flag
        """
        ...

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration

        Returns:
            Current configuration
        """
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """
    Interface for components that emit events.
    """

    def add_event_listener(self, event_type: str, listener: callable) -> str:
        """
        Add an event listener

        Args:
            event_type: Type of event to listen for
            listener: Listener function

        Returns:
            Listener ID
        """
        ...

    def remove_event_listener(self, listener_id: str) -> bool:
        """
        Remove an event listener

        Args:
            listener_id: ID of listener to remove

        Returns:
            Success flag
        """
        ...

    def emit_event(self, event_type: str, event_data: Any) -> None:
        """
        Emit an event

        Args:
            event_type: Type of event
            event_data: Event data
        """
        ...


@runtime_checkable
class CTASIntegrated(Protocol):
    """
    Interface for components that integrate with CTAS.
    """

    def get_ctas_task_ids(self) -> List[str]:
        """
        Get the CTAS task IDs this component implements

        Returns:
            List of CTAS task IDs
        """
        ...

    def get_ctas_support_level(self, task_id: str) -> float:
        """
        Get the component's support level for a CTAS task

        Args:
            task_id: CTAS task ID

        Returns:
            Support level (0.0 to 1.0)
        """
        ...

    def get_ctas_dependencies(self) -> Dict[str, List[str]]:
        """
        Get the component's CTAS dependencies

        Returns:
            Dictionary mapping task IDs to lists of dependency task IDs
        """
        ...


def is_nyxtrace_component(obj: object) -> TypeGuard[NyxTraceComponent]:
    """
    Check if an object implements the NyxTraceComponent interface

    Args:
        obj: Object to check

    Returns:
        True if the object implements NyxTraceComponent, False otherwise
    """
    return (
        hasattr(obj, "get_id")
        and callable(obj.get_id)
        and hasattr(obj, "get_metadata")
        and callable(obj.get_metadata)
        and hasattr(obj, "get_activation_parameters")
        and callable(obj.get_activation_parameters)
        and hasattr(obj, "is_activated")
        and callable(obj.is_activated)
    )


def is_registrable(obj: object) -> TypeGuard[Registrable]:
    """
    Check if an object implements the Registrable interface

    Args:
        obj: Object to check

    Returns:
        True if the object implements Registrable, False otherwise
    """
    return (
        hasattr(obj, "register")
        and callable(obj.register)
        and hasattr(obj, "unregister")
        and callable(obj.unregister)
    )


def is_lifecycle_managed(obj: object) -> TypeGuard[LifecycleManaged]:
    """
    Check if an object implements the LifecycleManaged interface

    Args:
        obj: Object to check

    Returns:
        True if the object implements LifecycleManaged, False otherwise
    """
    return (
        hasattr(obj, "initialize")
        and callable(obj.initialize)
        and hasattr(obj, "shutdown")
        and callable(obj.shutdown)
    )


def is_configurable(obj: object) -> TypeGuard[Configurable]:
    """
    Check if an object implements the Configurable interface

    Args:
        obj: Object to check

    Returns:
        True if the object implements Configurable, False otherwise
    """
    return (
        hasattr(obj, "configure")
        and callable(obj.configure)
        and hasattr(obj, "get_configuration")
        and callable(obj.get_configuration)
    )


def is_event_emitter(obj: object) -> TypeGuard[EventEmitter]:
    """
    Check if an object implements the EventEmitter interface

    Args:
        obj: Object to check

    Returns:
        True if the object implements EventEmitter, False otherwise
    """
    return (
        hasattr(obj, "add_event_listener")
        and callable(obj.add_event_listener)
        and hasattr(obj, "remove_event_listener")
        and callable(obj.remove_event_listener)
        and hasattr(obj, "emit_event")
        and callable(obj.emit_event)
    )


def is_ctas_integrated(obj: object) -> TypeGuard[CTASIntegrated]:
    """
    Check if an object implements the CTASIntegrated interface

    Args:
        obj: Object to check

    Returns:
        True if the object implements CTASIntegrated, False otherwise
    """
    return (
        hasattr(obj, "get_ctas_task_ids")
        and callable(obj.get_ctas_task_ids)
        and hasattr(obj, "get_ctas_support_level")
        and callable(obj.get_ctas_support_level)
        and hasattr(obj, "get_ctas_dependencies")
        and callable(obj.get_ctas_dependencies)
    )
