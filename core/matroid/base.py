"""
Matroid Base Module
-----------------
Implements the core matroid theory concepts used in the NyxTrace system.
Matroids are used to model task independence, convergent intelligence analysis,
and resource optimization in the CTAS framework.
"""

from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    List,
    Set,
    Tuple,
    Optional,
    TypeVar,
    Generic,
    FrozenSet,
    Callable,
)
import itertools
import uuid
import logging
import json

logger = logging.getLogger(__name__)

T = TypeVar("T")
Element = TypeVar("Element")


class Matroid(Generic[Element]):
    """
    Base class for matroid implementations.

    A matroid is a mathematical structure that generalizes the concept of linear independence
    in vector spaces. In the context of NyxTrace, matroids are used to model task independence,
    convergent intelligence analysis, and resource optimization.

    This class provides basic matroid operations based on independent sets.
    """

    def __init__(self, name: str = None):
        """
        Initialize a matroid

        Args:
            name: Optional name for the matroid
        """
        self.name = name or f"matroid-{uuid.uuid4()}"
        self._ground_set: Set[Element] = set()
        self._independent_sets: List[Set[Element]] = []

    def add_to_ground_set(self, element: Element) -> None:
        """
        Add an element to the ground set

        Args:
            element: Element to add
        """
        self._ground_set.add(element)
        # The empty set is always independent
        if not self._independent_sets:
            self._independent_sets.append(set())

        # Add singleton set if independent
        singleton = {element}
        if (
            self.is_independent(singleton)
            and singleton not in self._independent_sets
        ):
            self._independent_sets.append(singleton)

    def add_independent_set(self, independent_set: Set[Element]) -> None:
        """
        Add an independent set

        Args:
            independent_set: Independent set to add

        Raises:
            ValueError: If the set violates matroid properties
        """
        # Check that all elements are in the ground set
        if not independent_set.issubset(self._ground_set):
            raise ValueError(
                "Independent set contains elements not in the ground set"
            )

        # Check hereditary property
        for existing_set in self._independent_sets:
            # If this is a subset of an existing independent set, it's redundant
            if independent_set.issubset(existing_set):
                return

            # If this is a superset of an existing independent set,
            # check that it's actually independent
            if existing_set.issubset(
                independent_set
            ) and not self.is_independent(independent_set):
                raise ValueError(
                    "Adding this set would violate the hereditary property of matroids"
                )

        self._independent_sets.append(independent_set)

        # Add all subsets to maintain hereditary property
        for size in range(1, len(independent_set)):
            for subset in itertools.combinations(independent_set, size):
                subset_set = set(subset)
                if subset_set not in self._independent_sets:
                    self._independent_sets.append(subset_set)

    def get_ground_set(self) -> Set[Element]:
        """
        Get the ground set

        Returns:
            The ground set
        """
        return self._ground_set.copy()

    def get_independent_sets(self) -> List[Set[Element]]:
        """
        Get all independent sets

        Returns:
            List of independent sets
        """
        return [s.copy() for s in self._independent_sets]

    def is_independent(self, subset: Set[Element]) -> bool:
        """
        Check if a subset is independent

        Args:
            subset: Subset to check

        Returns:
            True if the subset is independent, False otherwise
        """
        # Empty set is always independent
        if not subset:
            return True

        # Check that all elements are in the ground set
        if not subset.issubset(self._ground_set):
            return False

        # Check if the set is already known to be independent
        for independent_set in self._independent_sets:
            if subset <= independent_set:  # subset.issubset(independent_set)
                return True

        # Otherwise, we need to check independence
        # This should be implemented in subclasses if needed
        return self._check_independence(subset)

    def _check_independence(self, subset: Set[Element]) -> bool:
        """
        Check if a subset is independent (implementation-specific)

        Args:
            subset: Subset to check

        Returns:
            True if the subset is independent, False otherwise
        """
        # Default implementation always returns False for unknown sets
        return False

    def rank(self, subset: Optional[Set[Element]] = None) -> int:
        """
        Calculate the rank of a subset

        Args:
            subset: Subset to calculate rank for, or None for the whole matroid

        Returns:
            Rank of the subset
        """
        if subset is None:
            subset = self._ground_set

        if not subset:
            return 0

        # The rank is the size of the largest independent subset
        max_size = 0
        for independent_set in self._independent_sets:
            if (
                independent_set.issubset(subset)
                and len(independent_set) > max_size
            ):
                max_size = len(independent_set)

        return max_size

    def find_basis(self, subset: Optional[Set[Element]] = None) -> Set[Element]:
        """
        Find a basis (maximal independent subset) for a subset

        Args:
            subset: Subset to find basis for, or None for the whole matroid

        Returns:
            A basis for the subset
        """
        if subset is None:
            subset = self._ground_set

        # Apply the greedy algorithm
        basis = set()

        # Sort elements for deterministic behavior
        sorted_elements = sorted(subset, key=str)

        for element in sorted_elements:
            test_set = basis.union({element})
            if self.is_independent(test_set):
                basis.add(element)

        return basis

    def find_circuits(self) -> List[Set[Element]]:
        """
        Find all circuits in the matroid

        A circuit is a minimal dependent set, meaning that removing any element
        makes it independent.

        Returns:
            List of circuits
        """
        circuits = []

        # Check all subsets of size at least 1
        for size in range(2, len(self._ground_set) + 1):
            for subset in itertools.combinations(self._ground_set, size):
                subset_set = set(subset)

                # Check if dependent
                if not self.is_independent(subset_set):
                    # Check if minimal
                    is_minimal = True
                    for element in subset_set:
                        if not self.is_independent(subset_set - {element}):
                            is_minimal = False
                            break

                    if is_minimal:
                        circuits.append(subset_set)

        return circuits

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert matroid to dictionary representation

        Returns:
            Dictionary representation
        """
        # Convert sets to lists for JSON serialization
        elements = list(map(str, self._ground_set))
        independent_sets = [list(map(str, s)) for s in self._independent_sets]

        return {
            "name": self.name,
            "ground_set": elements,
            "independent_sets": independent_sets,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Matroid":
        """
        Create matroid from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            Matroid instance
        """
        matroid = cls(name=data.get("name"))

        # Add elements to ground set
        for element in data.get("ground_set", []):
            matroid.add_to_ground_set(element)

        # Add independent sets
        for independent_set in data.get("independent_sets", []):
            matroid.add_independent_set(set(independent_set))

        return matroid

    def __str__(self) -> str:
        """String representation"""
        return f"Matroid(name={self.name}, ground_set={len(self._ground_set)} elements, independent_sets={len(self._independent_sets)})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"Matroid(name={self.name}, ground_set={self._ground_set}, independent_sets={self._independent_sets})"


class RankMatroid(Matroid[Element]):
    """
    Matroid defined by a rank function.

    A rank function r must satisfy:
    1. 0 <= r(X) <= |X| for all X
    2. If X ⊆ Y, then r(X) <= r(Y) (monotonicity)
    3. If X and Y are sets, then r(X ∪ Y) + r(X ∩ Y) <= r(X) + r(Y) (submodularity)
    """

    def __init__(
        self, rank_function: Callable[[Set[Element]], int], name: str = None
    ):
        """
        Initialize a rank matroid

        Args:
            rank_function: Function that computes the rank of a set
            name: Optional name for the matroid
        """
        super().__init__(name)
        self._rank_function = rank_function

    def _check_independence(self, subset: Set[Element]) -> bool:
        """
        Check if a subset is independent using the rank function

        A set X is independent if and only if r(X) = |X|

        Args:
            subset: Subset to check

        Returns:
            True if the subset is independent, False otherwise
        """
        return self._rank_function(subset) == len(subset)

    def rank(self, subset: Optional[Set[Element]] = None) -> int:
        """
        Calculate the rank of a subset

        Args:
            subset: Subset to calculate rank for, or None for the whole matroid

        Returns:
            Rank of the subset
        """
        if subset is None:
            subset = self._ground_set

        return self._rank_function(subset)


class IndependenceMatroid(Matroid[Element]):
    """
    Matroid defined by an independence oracle.

    An independence oracle is a function that returns True if a set is independent
    and False otherwise. The function must satisfy the matroid axioms.
    """

    def __init__(
        self,
        independence_oracle: Callable[[Set[Element]], bool],
        name: str = None,
    ):
        """
        Initialize an independence matroid

        Args:
            independence_oracle: Function that determines if a set is independent
            name: Optional name for the matroid
        """
        super().__init__(name)
        self._independence_oracle = independence_oracle

    def _check_independence(self, subset: Set[Element]) -> bool:
        """
        Check if a subset is independent using the independence oracle

        Args:
            subset: Subset to check

        Returns:
            True if the subset is independent, False otherwise
        """
        return self._independence_oracle(subset)


class TaskMatroid(Matroid[str]):
    """
    Matroid specifically designed for modeling task independence in the CTAS framework.

    Tasks are independent if they can be performed simultaneously without conflicts.
    This matroid helps optimize resource allocation and task scheduling.
    """

    def __init__(
        self,
        name: str = None,
        resource_constraints: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize a task matroid

        Args:
            name: Optional name for the matroid
            resource_constraints: Dictionary mapping resource types to dictionaries
                mapping task IDs to resource requirements
        """
        super().__init__(name)
        self._resource_constraints = resource_constraints or {}
        self._task_dependencies: Dict[str, Set[str]] = (
            {}
        )  # task -> set of prerequisites

    def add_task(self, task_id: str) -> None:
        """
        Add a task to the ground set

        Args:
            task_id: Task identifier
        """
        self.add_to_ground_set(task_id)

    def add_resource_constraint(
        self, resource_type: str, task_id: str, requirement: float
    ) -> None:
        """
        Add a resource constraint for a task

        Args:
            resource_type: Type of resource
            task_id: Task identifier
            requirement: Amount of resource required
        """
        if resource_type not in self._resource_constraints:
            self._resource_constraints[resource_type] = {}

        self._resource_constraints[resource_type][task_id] = requirement

    def add_task_dependency(self, task_id: str, prerequisite_id: str) -> None:
        """
        Add a task dependency

        Args:
            task_id: Task identifier
            prerequisite_id: Prerequisite task identifier
        """
        if task_id not in self._task_dependencies:
            self._task_dependencies[task_id] = set()

        self._task_dependencies[task_id].add(prerequisite_id)

    def _check_independence(self, subset: Set[str]) -> bool:
        """
        Check if a subset of tasks is independent

        Tasks are independent if:
        1. No task depends on another task not in the set
        2. Resource constraints are satisfied

        Args:
            subset: Subset of tasks to check

        Returns:
            True if the subset is independent, False otherwise
        """
        # Check dependencies
        for task_id in subset:
            if task_id in self._task_dependencies:
                for prerequisite_id in self._task_dependencies[task_id]:
                    if prerequisite_id not in subset:
                        return False

        # Check resource constraints
        for resource_type, requirements in self._resource_constraints.items():
            total_required = sum(
                requirements.get(task_id, 0)
                for task_id in subset
                if task_id in requirements
            )

            # For now, assume each resource has capacity 1.0
            # This could be extended to support different capacities
            if total_required > 1.0:
                return False

        return True

    def optimize_allocation(
        self,
        priorities: Dict[str, float],
        available_tasks: Optional[Set[str]] = None,
    ) -> List[Set[str]]:
        """
        Find an optimal allocation of tasks based on priorities

        Args:
            priorities: Task priorities (higher is more important)
            available_tasks: Set of available tasks, or None for all tasks

        Returns:
            List of task sets, where each set can be executed in parallel
        """
        if available_tasks is None:
            available_tasks = self._ground_set

        # Sort tasks by priority
        sorted_tasks = sorted(
            available_tasks,
            key=lambda task_id: priorities.get(task_id, 0.0),
            reverse=True,
        )

        result = []
        remaining_tasks = set(sorted_tasks)

        while remaining_tasks:
            # Find the largest independent subset of remaining tasks
            current_set = set()

            for task_id in sorted_tasks:
                if task_id in remaining_tasks:
                    test_set = current_set.union({task_id})
                    if self.is_independent(test_set):
                        current_set.add(task_id)

            # Add the set to the result
            result.append(current_set)

            # Remove these tasks from remaining tasks
            remaining_tasks -= current_set

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task matroid to dictionary representation

        Returns:
            Dictionary representation
        """
        base_dict = super().to_dict()

        # Add task-specific information
        task_dependencies = {
            task_id: list(prerequisites)
            for task_id, prerequisites in self._task_dependencies.items()
        }

        base_dict.update(
            {
                "resource_constraints": self._resource_constraints,
                "task_dependencies": task_dependencies,
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMatroid":
        """
        Create task matroid from dictionary representation

        Args:
            data: Dictionary representation

        Returns:
            TaskMatroid instance
        """
        matroid = cls(
            name=data.get("name"),
            resource_constraints=data.get("resource_constraints", {}),
        )

        # Add elements to ground set
        for element in data.get("ground_set", []):
            matroid.add_to_ground_set(element)

        # Add task dependencies
        for task_id, prerequisites in data.get("task_dependencies", {}).items():
            for prerequisite_id in prerequisites:
                matroid.add_task_dependency(task_id, prerequisite_id)

        return matroid
