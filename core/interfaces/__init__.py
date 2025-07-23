"""
NyxTrace Interfaces Package
--------------------------
Core interfaces for the NyxTrace platform.
"""

from core.interfaces.base import (
    NyxTraceComponent,
    Registrable,
    LifecycleManaged,
    Configurable,
    EventEmitter,
    CTASIntegrated,
)

from core.interfaces.collectors import EEI, CollectorParams, Collector

from core.interfaces.processors import (
    ProcessedIntelligence,
    ProcessorParams,
    Processor,
)

from core.interfaces.visualizers import (
    VisualizerParams,
    VisualizationOutput,
    Visualizer,
)

# Export all interfaces
__all__ = [
    # Base interfaces
    "NyxTraceComponent",
    "Registrable",
    "LifecycleManaged",
    "Configurable",
    "EventEmitter",
    "CTASIntegrated",
    # Collector interfaces
    "EEI",
    "CollectorParams",
    "Collector",
    # Processor interfaces
    "ProcessedIntelligence",
    "ProcessorParams",
    "Processor",
    # Visualizer interfaces
    "VisualizerParams",
    "VisualizationOutput",
    "Visualizer",
]
