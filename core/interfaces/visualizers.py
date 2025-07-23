"""
NyxTrace Visualizer Interfaces
---------------------------
Interfaces for data visualization components.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union

from core.interfaces.base import NyxTraceComponent, Registrable


@dataclass
class VisualizerParams:
    """
    Parameters for visualization operations.

    This class defines the parameters for a visualization
    operation, including options and custom parameters.
    """

    options: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationOutput:
    """
    Output of a visualization operation.

    This class represents the result of a visualization
    operation, including the rendered content and metadata.
    """

    visualization_id: str
    visualizer_id: str
    render_time: datetime.datetime
    content: Any
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Visualizer(NyxTraceComponent, Registrable):
    """
    Interface for data visualizers.

    Data visualizers are responsible for rendering data
    in a format suitable for human consumption.
    """

    def visualize(
        self, data: Any, params: Optional[VisualizerParams] = None
    ) -> None:
        """
        Visualize data

        Args:
            data: Data to visualize
            params: Visualization parameters
        """
        raise NotImplementedError()

    def get_visualization_domain(self) -> str:
        """Get the domain this visualizer operates in"""
        raise NotImplementedError()

    def get_visualization_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this visualizer"""
        raise NotImplementedError()
