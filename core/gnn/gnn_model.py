"""
Graph Neural Network Model Module
-------------------------------
Implements GNN models for CTAS task analysis and intelligence fusion.
Integrates with the matroid framework for task independence modeling.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
import os
import json
import numpy as np

# Try importing torch and torch_geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class CTASGraphData:
    """
    Container for CTAS graph data.
    Abstracts the underlying graph library and provides conversion utilities.
    """

    def __init__(self):
        """Initialize a CTASGraphData instance"""
        self.node_ids = []  # UUID/SCH IDs
        self.node_features = []  # Feature vectors
        self.edge_index = []  # Connectivity
        self.edge_attr = []  # Edge attributes
        self.labels = []  # Node labels if any

    def add_node(
        self, node_id: str, features: List[float], label: Optional[Any] = None
    ) -> int:
        """
        Add a node to the graph

        Args:
            node_id: Node identifier (UUID/SCH)
            features: Feature vector
            label: Optional node label

        Returns:
            Node index
        """
        node_idx = len(self.node_ids)
        self.node_ids.append(node_id)
        self.node_features.append(features)
        self.labels.append(label)
        return node_idx

    def add_edge(
        self, src_idx: int, dst_idx: int, attr: Optional[List[float]] = None
    ) -> None:
        """
        Add an edge to the graph

        Args:
            src_idx: Source node index
            dst_idx: Destination node index
            attr: Optional edge attributes
        """
        self.edge_index.append((src_idx, dst_idx))
        self.edge_attr.append(attr or [])

    def add_edge_by_id(
        self, src_id: str, dst_id: str, attr: Optional[List[float]] = None
    ) -> bool:
        """
        Add an edge by node identifiers

        Args:
            src_id: Source node identifier
            dst_id: Destination node identifier
            attr: Optional edge attributes

        Returns:
            True if the edge was added, False if nodes not found
        """
        try:
            src_idx = self.node_ids.index(src_id)
            dst_idx = self.node_ids.index(dst_id)
            self.add_edge(src_idx, dst_idx, attr)
            return True
        except ValueError:
            logger.warning(
                f"Could not add edge {src_id} -> {dst_id}: Node not found"
            )
            return False

    def get_node_index(self, node_id: str) -> int:
        """
        Get the index of a node

        Args:
            node_id: Node identifier

        Returns:
            Node index

        Raises:
            ValueError: If node not found
        """
        return self.node_ids.index(node_id)

    def get_node_id(self, node_idx: int) -> str:
        """
        Get the identifier of a node

        Args:
            node_idx: Node index

        Returns:
            Node identifier

        Raises:
            IndexError: If index out of range
        """
        return self.node_ids[node_idx]

    def get_node_data(self, node_id: str) -> Tuple[List[float], Any]:
        """
        Get the data for a node

        Args:
            node_id: Node identifier

        Returns:
            Tuple of (features, label)

        Raises:
            ValueError: If node not found
        """
        idx = self.get_node_index(node_id)
        return self.node_features[idx], self.labels[idx]

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get the neighbors of a node

        Args:
            node_id: Node identifier

        Returns:
            List of neighbor identifiers

        Raises:
            ValueError: If node not found
        """
        idx = self.get_node_index(node_id)
        neighbors = []

        for src, dst in self.edge_index:
            if src == idx:
                neighbors.append(self.node_ids[dst])
            elif dst == idx:  # Undirected graph
                neighbors.append(self.node_ids[src])

        return neighbors

    def to_torch_geometric(self) -> Optional["Data"]:
        """
        Convert to a PyTorch Geometric Data object

        Returns:
            PyTorch Geometric Data or None if torch_geometric not available
        """
        if not HAS_TORCH:
            logger.warning(
                "torch and torch_geometric are required for GNN models"
            )
            return None

        # Convert node features to tensor
        x = torch.tensor(self.node_features, dtype=torch.float)

        # Convert edge index to tensor
        edge_index = torch.tensor(
            list(zip(*self.edge_index)), dtype=torch.long  # Transpose
        )

        # Convert edge attributes to tensor if present
        edge_attr = None
        if self.edge_attr and all(attr for attr in self.edge_attr):
            edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)

        # Convert labels to tensor if present
        y = None
        if self.labels and all(label is not None for label in self.labels):
            try:
                y = torch.tensor(self.labels, dtype=torch.float)
            except (TypeError, ValueError):
                # Non-numeric labels
                logger.warning(
                    "Non-numeric labels cannot be converted to tensor"
                )

        # Create Data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation

        Returns:
            Dictionary representation
        """
        return {
            "node_ids": self.node_ids,
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CTASGraphData":
        """
        Create from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            CTASGraphData instance
        """
        graph = cls()
        graph.node_ids = data.get("node_ids", [])
        graph.node_features = data.get("node_features", [])
        graph.edge_index = data.get("edge_index", [])
        graph.edge_attr = data.get("edge_attr", [])
        graph.labels = data.get("labels", [])
        return graph

    def __len__(self) -> int:
        """Get the number of nodes"""
        return len(self.node_ids)


class CTASGraphBuilder:
    """
    Builder for constructing CTAS graphs from task nodes and matroids.
    """

    def __init__(self):
        """Initialize a CTASGraphBuilder instance"""
        self.graph = CTASGraphData()

    def add_task_node(self, task_id: str, task_data: Dict[str, Any]) -> int:
        """
        Add a task node to the graph

        Args:
            task_id: Task identifier (UUID/SCH)
            task_data: Task data with features

        Returns:
            Node index
        """
        # Extract node features from task data
        features = self._extract_task_features(task_data)

        # Add node
        return self.graph.add_node(
            task_id, features, label=task_data.get("task_name")
        )

    def add_task_relationship(
        self, src_id: str, dst_id: str, relationship_type: str
    ) -> bool:
        """
        Add a relationship between tasks

        Args:
            src_id: Source task identifier
            dst_id: Destination task identifier
            relationship_type: Type of relationship

        Returns:
            True if the relationship was added, False otherwise
        """
        # Convert relationship type to one-hot encoding
        rel_types = [
            "REQUIRES",
            "SUPPORTS",
            "FOLLOWS",
            "COMES_BEFORE",
            "IS_PART_OF",
        ]
        attr = [1.0 if t == relationship_type else 0.0 for t in rel_types]

        return self.graph.add_edge_by_id(src_id, dst_id, attr)

    def build_from_tasks(self, tasks: List[Dict[str, Any]]) -> CTASGraphData:
        """
        Build a graph from task data

        Args:
            tasks: List of task data dictionaries

        Returns:
            CTASGraphData instance
        """
        # Reset graph
        self.graph = CTASGraphData()

        # Add all nodes
        for task in tasks:
            task_id = task.get("task_id")
            if task_id:
                self.add_task_node(task_id, task)

        # Add edges based on relationships
        for task in tasks:
            task_id = task.get("task_id")
            if not task_id:
                continue

            # Add relationships if present
            relationships = task.get("relationships", {})

            for rel_type, targets in relationships.items():
                if isinstance(targets, list):
                    for target_id in targets:
                        self.add_task_relationship(task_id, target_id, rel_type)
                elif isinstance(targets, str):
                    self.add_task_relationship(task_id, targets, rel_type)

        return self.graph

    def build_from_matroid(
        self, matroid, task_data: Dict[str, Dict[str, Any]]
    ) -> CTASGraphData:
        """
        Build a graph from a matroid and task data

        Args:
            matroid: A matroid instance
            task_data: Dictionary mapping task IDs to task data

        Returns:
            CTASGraphData instance
        """
        # Reset graph
        self.graph = CTASGraphData()

        # Add all nodes
        for element in matroid.get_ground_set():
            element_str = str(element)
            if element_str in task_data:
                self.add_task_node(element_str, task_data[element_str])
            else:
                # Create a basic node if no task data
                self.graph.add_node(element_str, [0.0] * 5)

        # Add independent sets as special edges
        for ind_set in matroid.get_independent_sets():
            if len(ind_set) >= 2:
                # Create edges for this independent set
                elements = list(ind_set)
                for i in range(len(elements)):
                    for j in range(i + 1, len(elements)):
                        self.graph.add_edge_by_id(
                            str(elements[i]),
                            str(elements[j]),
                            [
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ],  # INDEPENDENT relationship
                        )

        return self.graph

    def _extract_task_features(self, task_data: Dict[str, Any]) -> List[float]:
        """
        Extract feature vector from task data

        Args:
            task_data: Task data dictionary

        Returns:
            Feature vector
        """
        # Extract standard features
        features = []

        # P value (probability)
        p_value = task_data.get("P", 0.5)
        if isinstance(p_value, str):
            try:
                p_value = float(p_value)
            except ValueError:
                p_value = 0.5
        features.append(float(p_value))

        # P' value (adjusted probability)
        p_prime = task_data.get("P'", 0.5)
        if isinstance(p_prime, str):
            try:
                p_prime = float(p_prime)
            except ValueError:
                p_prime = 0.5
        features.append(float(p_prime))

        # Entropy (ζ)
        entropy = task_data.get("entropy", 0.5)
        if isinstance(entropy, str):
            try:
                entropy = float(entropy)
            except ValueError:
                entropy = 0.5
        features.append(float(entropy))

        # κ value (plasticity or complexity)
        kappa = task_data.get("kappa", 0.5)
        if isinstance(kappa, str):
            try:
                kappa = float(kappa)
            except ValueError:
                kappa = 0.5
        features.append(float(kappa))

        # τ value (transition readiness)
        tau = task_data.get("transition_readiness", 0.5)
        if isinstance(tau, str):
            try:
                tau = float(tau)
            except ValueError:
                tau = 0.5
        features.append(float(tau))

        # Symbol flag (boolean indicator)
        symbol_flag = 1.0 if task_data.get("symbol_flag", False) else 0.0
        features.append(float(symbol_flag))

        # We could add semantic vector here if available
        semantic_vector = task_data.get("semantic_vector", [])
        if isinstance(semantic_vector, list) and len(semantic_vector) > 0:
            features.extend(
                [float(v) for v in semantic_vector[:10]]
            )  # Include up to 10 dimensions

        return features


class GNNModel:
    """
    Graph Neural Network model for CTAS task analysis.
    Implements various GNN architectures depending on available libraries.
    """

    def __init__(
        self,
        model_type: str = "GCN",
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
    ):
        """
        Initialize a GNN model

        Args:
            model_type: Type of GNN ("GCN", "GAT", "GraphSAGE")
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of message passing layers
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.model = None
        self.initialized = False

        # Initialize if torch is available
        if HAS_TORCH:
            self._initialize_model()
        else:
            logger.warning(
                "torch and torch_geometric are required for GNN models"
            )

    def _initialize_model(self) -> None:
        """Initialize the GNN model"""
        if not HAS_TORCH:
            return

        # Choose model type
        if self.model_type == "GCN":
            self.model = GCNNetwork(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                self.num_layers,
            )
        elif self.model_type == "GAT":
            self.model = GATNetwork(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                self.num_layers,
            )
        elif self.model_type == "GraphSAGE":
            self.model = GraphSAGENetwork(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                self.num_layers,
            )
        else:
            logger.warning(
                f"Unknown model type: {self.model_type}. Using GCN instead."
            )
            self.model = GCNNetwork(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                self.num_layers,
            )

        self.initialized = True

    def encode_graph(
        self, graph_data: CTASGraphData
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Encode a graph into node embeddings

        Args:
            graph_data: Graph data

        Returns:
            Dictionary mapping node IDs to embedding vectors
        """
        if not self.initialized or not HAS_TORCH:
            logger.warning("Model not initialized or torch not available")
            return None

        # Convert to PyTorch Geometric Data
        data = graph_data.to_torch_geometric()
        if data is None:
            return None

        # Switch to evaluation mode
        self.model.eval()

        # Process through the model
        with torch.no_grad():
            node_embeddings = self.model(data.x, data.edge_index)

        # Convert to numpy arrays
        embeddings = node_embeddings.cpu().numpy()

        # Create result dictionary
        result = {}
        for i, node_id in enumerate(graph_data.node_ids):
            result[node_id] = embeddings[i]

        return result

    def save_model(self, path: str) -> bool:
        """
        Save the model to a file

        Args:
            path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not HAS_TORCH:
            logger.warning("Model not initialized or torch not available")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save model state
            torch.save(self.model.state_dict(), path)

            # Save model configuration
            config_path = path + ".config"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "model_type": self.model_type,
                        "input_dim": self.input_dim,
                        "hidden_dim": self.hidden_dim,
                        "output_dim": self.output_dim,
                        "num_layers": self.num_layers,
                    },
                    f,
                )

            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    @classmethod
    def load_model(cls, path: str) -> Optional["GNNModel"]:
        """
        Load a model from a file

        Args:
            path: Path to load the model from

        Returns:
            GNNModel instance or None if loading failed
        """
        if not HAS_TORCH:
            logger.warning(
                "torch and torch_geometric are required for GNN models"
            )
            return None

        try:
            # Load model configuration
            config_path = path + ".config"
            with open(config_path, "r") as f:
                config = json.load(f)

            # Create model
            model = cls(
                model_type=config["model_type"],
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                num_layers=config["num_layers"],
            )

            # Load state dict
            model.model.load_state_dict(torch.load(path))

            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
