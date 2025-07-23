"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-SIMULATED-ANNEALING-0001            │
// │ 📁 domain       : Mathematics, Optimization                 │
// │ 🧠 description  : Simulated annealing implementation for    │
// │                  complex optimization problems              │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_OPTIMIZATION                        │
// │ 🧩 dependencies : numpy, time, logging                     │
// │ 🔧 tool_usage   : Optimization, Search                     │
// │ 📡 input_type   : Objective functions                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : optimization, energy minimization         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Simulated Annealing Module
-----------------------
Implementation of simulated annealing for solving complex optimization
problems by simulating physical annealing processes. This algorithm
excels at finding global optima in rugged landscapes with many local minima.
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Generic,
)
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("optimization.simulated_annealing")
logger.setLevel(logging.INFO)

# Generic type for solution
T = TypeVar("T")


class CoolingSchedule(Enum):
    """
    Cooling schedule types for simulated annealing

    # Class defines subject schedules
    # Enumeration specifies predicate options
    # Type catalogs object variants
    """

    EXPONENTIAL = "exponential"  # T(k) = T0 * alpha^k
    LINEAR = "linear"  # T(k) = T0 * (1 - k/max_iter)
    LOGARITHMIC = "logarithmic"  # T(k) = T0 / log(k+2)
    BOLTZMANN = "boltzmann"  # T(k) = T0 / ln(k+1)
    CAUCHY = "cauchy"  # T(k) = T0 / (k+1)
    CUSTOM = "custom"  # User-defined function


@dataclass
class AnnealingResult(Generic[T]):
    """
    Result of simulated annealing optimization

    # Class stores subject results
    # Container holds predicate data
    # Structure formats object output
    """

    best_solution: T
    best_energy: float
    initial_energy: float
    iterations: int
    acceptance_rate: float
    temperature_history: List[float]
    energy_history: List[float]
    execution_time: float
    cooling_schedule: CoolingSchedule
    metadata: Dict[str, Any] = field(default_factory=dict)

    def improvement_percentage(self) -> float:
        """
        Calculate percentage improvement from initial to best solution

        # Function calculates subject improvement
        # Method quantifies predicate progress
        # Operation computes object efficiency

        Returns:
            Percentage improvement (positive for minimization problems)
        """
        if self.initial_energy == 0:
            return 0.0

        improvement = self.initial_energy - self.best_energy
        return (improvement / abs(self.initial_energy)) * 100

    def plot_energy_history(
        self, show_temp: bool = True, filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot energy history during annealing process

        # Function plots subject history
        # Method visualizes predicate convergence
        # Operation displays object progress

        Args:
            show_temp: Whether to show temperature on second axis
            filename: If provided, save plot to this file

        Returns:
            Matplotlib figure object
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot energy history
        ax1.plot(self.energy_history, "b-", label="Energy")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Energy", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Add temperature on second axis if requested
        if show_temp:
            ax2 = ax1.twinx()
            ax2.plot(self.temperature_history, "r-", label="Temperature")
            ax2.set_ylabel("Temperature", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

        # Add title and grid
        plt.title(
            f"Simulated Annealing - {self.cooling_schedule.value} cooling"
        )
        ax1.grid(True, alpha=0.3)

        # Add text box with statistics
        stats_text = (
            f"Initial Energy: {self.initial_energy:.4f}\n"
            f"Final Energy: {self.best_energy:.4f}\n"
            f"Improvement: {self.improvement_percentage():.2f}%\n"
            f"Iterations: {self.iterations}\n"
            f"Acceptance Rate: {self.acceptance_rate:.2f}\n"
            f"Time: {self.execution_time:.2f}s"
        )

        plt.figtext(
            0.02,
            0.02,
            stats_text,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Create legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_temp:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        plt.tight_layout()

        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        return fig


@dataclass
class AnnealingConfig:
    """
    Configuration for simulated annealing

    # Class configures subject algorithm
    # Structure defines predicate parameters
    # Object specifies annealing settings
    """

    initial_temperature: float = 100.0
    final_temperature: float = 1e-6
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    cooling_factor: float = 0.95  # Alpha for exponential cooling
    max_iterations: int = 10000
    max_time_seconds: Optional[float] = None
    random_seed: Optional[int] = None
    equilibrium_iterations: int = 1  # Iterations at each temperature
    custom_cooling_function: Optional[Callable[[float, int, int], float]] = None
    reheat_trigger: Optional[float] = (
        None  # Trigger reheating if no improvement after this fraction of iterations
    )
    reheat_factor: float = 2.0  # Factor to multiply temperature when reheating
    max_reheats: int = 3  # Maximum number of reheats
    parallel: bool = False
    max_workers: Optional[int] = None


class SimulatedAnnealing(Generic[T]):
    """
    Simulated annealing optimizer

    # Class optimizes subject problems
    # Annealer executes predicate search
    # Engine finds object solutions
    """

    def __init__(self, config: Optional[AnnealingConfig] = None):
        """
        Initialize simulated annealing optimizer

        # Function initializes subject optimizer
        # Method prepares predicate framework
        # Operation configures object parameters

        Args:
            config: Configuration for the algorithm
        """
        self.config = config or AnnealingConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        logger.info(
            f"Simulated annealing initialized with seed: {self.config.random_seed}"
        )

    def optimize(
        self,
        initial_solution: T,
        energy_function: Callable[[T], float],
        neighbor_function: Callable[[T, np.random.RandomState, float], T],
        acceptance_function: Optional[
            Callable[[float, float, float], float]
        ] = None,
    ) -> AnnealingResult[T]:
        """
        Run simulated annealing optimization

        # Function optimizes subject problem
        # Method executes predicate annealing
        # Operation finds object solution

        Args:
            initial_solution: Initial candidate solution
            energy_function: Function to calculate solution energy (lower is better)
            neighbor_function: Function to generate neighboring solution
            acceptance_function: Function to determine acceptance probability
                                 (defaults to Metropolis criterion)

        Returns:
            AnnealingResult containing optimization results
        """
        start_time = time.time()

        # Initialize tracking variables
        current_solution = initial_solution
        current_energy = energy_function(current_solution)
        initial_energy = current_energy

        best_solution = current_solution
        best_energy = current_energy

        temperature = self.config.initial_temperature
        iterations = 0
        accepted_moves = 0

        # Initialize history tracking
        temperature_history = [temperature]
        energy_history = [current_energy]

        # Initialize reheat tracking
        reheat_count = 0
        last_improvement_iteration = 0

        logger.info(
            f"Starting simulated annealing with initial energy: {initial_energy}"
        )

        # Run optimization with parallel or sequential approach
        if self.config.parallel and self.config.equilibrium_iterations > 1:
            result = self._optimize_parallel(
                initial_solution,
                energy_function,
                neighbor_function,
                acceptance_function,
            )
        else:
            while True:
                # Check stopping conditions
                if iterations >= self.config.max_iterations:
                    logger.info(f"Reached maximum iterations: {iterations}")
                    break

                if (
                    self.config.max_time_seconds
                    and time.time() - start_time > self.config.max_time_seconds
                ):
                    logger.info(
                        f"Reached maximum time: {self.config.max_time_seconds} seconds"
                    )
                    break

                if temperature < self.config.final_temperature:
                    logger.info(f"Reached final temperature: {temperature}")
                    break

                # Perform equilibrium iterations at current temperature
                for _ in range(self.config.equilibrium_iterations):
                    # Generate a neighboring solution
                    neighbor = neighbor_function(
                        current_solution, self.rng, temperature
                    )
                    neighbor_energy = energy_function(neighbor)

                    # Determine whether to accept the neighbor
                    delta_energy = neighbor_energy - current_energy

                    # Use provided acceptance function or default to Metropolis
                    if acceptance_function:
                        acceptance_probability = acceptance_function(
                            current_energy, neighbor_energy, temperature
                        )
                    else:
                        # Metropolis criterion (always accept better solutions)
                        acceptance_probability = (
                            1.0
                            if delta_energy < 0
                            else np.exp(-delta_energy / temperature)
                        )

                    # Accept or reject the neighbor
                    if self.rng.random() < acceptance_probability:
                        current_solution = neighbor
                        current_energy = neighbor_energy
                        accepted_moves += 1

                        # Update best solution if improved
                        if current_energy < best_energy:
                            best_solution = current_solution
                            best_energy = current_energy
                            last_improvement_iteration = iterations

                    iterations += 1

                    # Check early stopping conditions
                    if iterations >= self.config.max_iterations:
                        break

                    if (
                        self.config.max_time_seconds
                        and time.time() - start_time
                        > self.config.max_time_seconds
                    ):
                        break

                # Update temperature according to cooling schedule
                temperature = self._update_temperature(temperature, iterations)

                # Track history
                temperature_history.append(temperature)
                energy_history.append(current_energy)

                # Check for reheat if configured
                if (
                    self.config.reheat_trigger
                    and iterations - last_improvement_iteration
                    > self.config.reheat_trigger * self.config.max_iterations
                    and reheat_count < self.config.max_reheats
                ):

                    # Reheat the system
                    old_temperature = temperature
                    temperature *= self.config.reheat_factor
                    reheat_count += 1

                    logger.info(
                        f"Reheating at iteration {iterations}: {old_temperature:.4f} -> {temperature:.4f} "
                        f"(no improvement for {iterations - last_improvement_iteration} iterations)"
                    )

                # Log progress
                if iterations % max(1, self.config.max_iterations // 10) == 0:
                    logger.info(
                        f"Iteration {iterations}/{self.config.max_iterations}, "
                        f"Temperature: {temperature:.6f}, Energy: {current_energy:.6f}"
                    )

            # Create result object
            result = AnnealingResult(
                best_solution=best_solution,
                best_energy=best_energy,
                initial_energy=initial_energy,
                iterations=iterations,
                acceptance_rate=accepted_moves / max(1, iterations),
                temperature_history=temperature_history,
                energy_history=energy_history,
                execution_time=time.time() - start_time,
                cooling_schedule=self.config.cooling_schedule,
                metadata={
                    "random_seed": self.config.random_seed,
                    "reheat_count": reheat_count,
                },
            )

        logger.info(
            f"Optimization complete: {result.best_energy:.6f} (improvement: {result.improvement_percentage():.2f}%)"
        )
        logger.info(
            f"Acceptance rate: {result.acceptance_rate:.2f}, Execution time: {result.execution_time:.2f}s"
        )

        return result

    def _optimize_parallel(
        self,
        initial_solution: T,
        energy_function: Callable[[T], float],
        neighbor_function: Callable[[T, np.random.RandomState, float], T],
        acceptance_function: Optional[
            Callable[[float, float, float], float]
        ] = None,
    ) -> AnnealingResult[T]:
        """
        Run simulated annealing optimization with parallel equilibrium iterations

        # Function optimizes subject problem
        # Method executes predicate parallel
        # Operation finds object solution

        Args:
            initial_solution: Initial candidate solution
            energy_function: Function to calculate solution energy
            neighbor_function: Function to generate neighboring solution
            acceptance_function: Function to determine acceptance probability

        Returns:
            AnnealingResult containing optimization results
        """
        start_time = time.time()

        # Initialize tracking variables
        current_solution = initial_solution
        current_energy = energy_function(current_solution)
        initial_energy = current_energy

        best_solution = current_solution
        best_energy = current_energy

        temperature = self.config.initial_temperature
        iterations = 0
        accepted_moves = 0

        # Initialize history tracking
        temperature_history = [temperature]
        energy_history = [current_energy]

        # Initialize reheat tracking
        reheat_count = 0
        last_improvement_iteration = 0

        logger.info(
            f"Starting parallel simulated annealing with initial energy: {initial_energy}"
        )

        # Define a worker function for parallel equilibrium iterations
        def worker_func(solution, energy, temp, seed, worker_id):
            worker_rng = np.random.RandomState(seed)
            local_solution = solution
            local_energy = energy
            local_best_solution = solution
            local_best_energy = energy
            local_accepted = 0

            # Perform a portion of equilibrium iterations
            for _ in range(local_iterations):
                # Generate a neighboring solution
                neighbor = neighbor_function(local_solution, worker_rng, temp)
                neighbor_energy = energy_function(neighbor)

                # Determine whether to accept the neighbor
                delta_energy = neighbor_energy - local_energy

                # Use provided acceptance function or default to Metropolis
                if acceptance_function:
                    acceptance_probability = acceptance_function(
                        local_energy, neighbor_energy, temp
                    )
                else:
                    # Metropolis criterion
                    acceptance_probability = (
                        1.0
                        if delta_energy < 0
                        else np.exp(-delta_energy / temp)
                    )

                # Accept or reject the neighbor
                if worker_rng.random() < acceptance_probability:
                    local_solution = neighbor
                    local_energy = neighbor_energy
                    local_accepted += 1

                    # Update local best solution if improved
                    if local_energy < local_best_energy:
                        local_best_solution = local_solution
                        local_best_energy = local_energy

            return (
                local_solution,
                local_energy,
                local_best_solution,
                local_best_energy,
                local_accepted,
            )

        # Iterate through cooling schedule
        while True:
            # Check stopping conditions
            if iterations >= self.config.max_iterations:
                logger.info(f"Reached maximum iterations: {iterations}")
                break

            if (
                self.config.max_time_seconds
                and time.time() - start_time > self.config.max_time_seconds
            ):
                logger.info(
                    f"Reached maximum time: {self.config.max_time_seconds} seconds"
                )
                break

            if temperature < self.config.final_temperature:
                logger.info(f"Reached final temperature: {temperature}")
                break

            # Distribute equilibrium iterations across workers
            worker_count = self.config.max_workers or 1
            total_eq_iterations = self.config.equilibrium_iterations
            local_iterations = max(1, total_eq_iterations // worker_count)

            # Use ProcessPoolExecutor for parallel equilibrium iterations
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                # Submit jobs for each worker
                futures = []
                for worker_id in range(worker_count):
                    # Generate different seed for each worker
                    worker_seed = (
                        None
                        if self.config.random_seed is None
                        else self.config.random_seed + iterations + worker_id
                    )

                    future = executor.submit(
                        worker_func,
                        current_solution,
                        current_energy,
                        temperature,
                        worker_seed,
                        worker_id,
                    )
                    futures.append(future)

                # Collect results
                local_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        local_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in worker: {e}")

            # Process worker results
            worker_accepted = 0
            best_worker_solution = None
            best_worker_energy = float("inf")

            for result in local_results:
                (
                    local_solution,
                    local_energy,
                    local_best,
                    local_best_energy,
                    local_accepted,
                ) = result
                worker_accepted += local_accepted

                # Track best solution across all workers
                if local_best_energy < best_worker_energy:
                    best_worker_solution = local_best
                    best_worker_energy = local_best_energy

            # Update current state with best result from workers
            if best_worker_energy < current_energy:
                current_solution = best_worker_solution
                current_energy = best_worker_energy

            # Update global best if improved
            if best_worker_energy < best_energy:
                best_solution = best_worker_solution
                best_energy = best_worker_energy
                last_improvement_iteration = iterations

            # Update tracking variables
            accepted_moves += worker_accepted
            iterations += total_eq_iterations

            # Update temperature according to cooling schedule
            temperature = self._update_temperature(temperature, iterations)

            # Track history
            temperature_history.append(temperature)
            energy_history.append(current_energy)

            # Check for reheat if configured
            if (
                self.config.reheat_trigger
                and iterations - last_improvement_iteration
                > self.config.reheat_trigger * self.config.max_iterations
                and reheat_count < self.config.max_reheats
            ):

                # Reheat the system
                old_temperature = temperature
                temperature *= self.config.reheat_factor
                reheat_count += 1

                logger.info(
                    f"Reheating at iteration {iterations}: {old_temperature:.4f} -> {temperature:.4f} "
                    f"(no improvement for {iterations - last_improvement_iteration} iterations)"
                )

            # Log progress
            if iterations % max(1, self.config.max_iterations // 10) == 0:
                logger.info(
                    f"Iteration {iterations}/{self.config.max_iterations}, "
                    f"Temperature: {temperature:.6f}, Energy: {current_energy:.6f}"
                )

        # Create result object
        return AnnealingResult(
            best_solution=best_solution,
            best_energy=best_energy,
            initial_energy=initial_energy,
            iterations=iterations,
            acceptance_rate=accepted_moves / max(1, iterations),
            temperature_history=temperature_history,
            energy_history=energy_history,
            execution_time=time.time() - start_time,
            cooling_schedule=self.config.cooling_schedule,
            metadata={
                "random_seed": self.config.random_seed,
                "parallel": True,
                "reheat_count": reheat_count,
            },
        )

    def _update_temperature(
        self, current_temperature: float, iteration: int
    ) -> float:
        """
        Update temperature according to cooling schedule

        # Function updates subject temperature
        # Method applies predicate schedule
        # Operation cools object system

        Args:
            current_temperature: Current temperature
            iteration: Current iteration number

        Returns:
            Updated temperature
        """
        schedule = self.config.cooling_schedule

        if schedule == CoolingSchedule.EXPONENTIAL:
            # T(k) = T0 * alpha^k
            return current_temperature * self.config.cooling_factor

        elif schedule == CoolingSchedule.LINEAR:
            # T(k) = T0 * (1 - k/max_iter)
            fraction = iteration / self.config.max_iterations
            return self.config.initial_temperature * (1 - min(1, fraction))

        elif schedule == CoolingSchedule.LOGARITHMIC:
            # T(k) = T0 / log(k+2)
            return self.config.initial_temperature / np.log(iteration + 2)

        elif schedule == CoolingSchedule.BOLTZMANN:
            # T(k) = T0 / ln(k+1)
            return self.config.initial_temperature / np.log(iteration + 2)

        elif schedule == CoolingSchedule.CAUCHY:
            # T(k) = T0 / (k+1)
            return self.config.initial_temperature / (iteration + 1)

        elif schedule == CoolingSchedule.CUSTOM:
            # User-defined cooling function
            if self.config.custom_cooling_function:
                return self.config.custom_cooling_function(
                    current_temperature, iteration, self.config.max_iterations
                )
            else:
                logger.warning(
                    "Custom cooling schedule selected but no function provided. Using exponential."
                )
                return current_temperature * self.config.cooling_factor

        else:
            # Default to exponential cooling
            return current_temperature * self.config.cooling_factor


# Example usage
def example_energy_function(x: List[float]) -> float:
    """
    Example energy function (Rosenbrock function)

    # Function computes subject energy
    # Method evaluates predicate cost
    # Operation calculates object value

    Args:
        x: Position in n-dimensional space

    Returns:
        Energy value (lower is better)
    """
    energy = 0
    for i in range(len(x) - 1):
        energy += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return energy


def example_neighbor_function(
    current: List[float],
    random_state: np.random.RandomState,
    temperature: float,
) -> List[float]:
    """
    Example neighbor function that perturbs the current solution

    # Function generates subject neighbor
    # Method perturbs predicate solution
    # Operation modifies object position

    Args:
        current: Current solution
        random_state: Random state for reproducibility
        temperature: Current temperature (can be used to scale perturbations)

    Returns:
        Neighboring solution
    """
    # Scale perturbation with temperature
    scale = temperature / 10.0

    # Create a copy to avoid modifying the original
    neighbor = current.copy()

    # Perturb each dimension
    for i in range(len(neighbor)):
        neighbor[i] += random_state.normal(0, scale)

    return neighbor
