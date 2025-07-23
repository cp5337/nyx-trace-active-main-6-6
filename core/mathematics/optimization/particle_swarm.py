"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-PARTICLE-SWARM-0001                 │
// │ 📁 domain       : Mathematics, Optimization                 │
// │ 🧠 description  : Particle Swarm Optimization (PSO)         │
// │                  for continuous optimization problems       │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_OPTIMIZATION                        │
// │ 🧩 dependencies : numpy, time, logging                     │
// │ 🔧 tool_usage   : Optimization, Search                     │
// │ 📡 input_type   : Objective functions                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : collective intelligence, swarm behavior   │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Particle Swarm Optimization Module
-------------------------------
Implementation of Particle Swarm Optimization (PSO) algorithm for
continuous optimization problems. PSO is inspired by social behavior
of bird flocking or fish schooling, using swarm intelligence to
optimize complex problems.
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("optimization.particle_swarm")
logger.setLevel(logging.INFO)


@dataclass
class Particle:
    """
    Particle representation for PSO algorithm

    # Class represents subject particle
    # Structure models predicate agent
    # Object encapsulates swarm element
    """

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    current_fitness: float


@dataclass
class PSOResult:
    """
    Result of particle swarm optimization

    # Class stores subject results
    # Container holds predicate data
    # Structure formats object output
    """

    best_position: np.ndarray
    best_fitness: float
    initial_best_fitness: float
    iterations: int
    swarm_size: int
    fitness_history: List[float]
    diversity_history: List[float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def improvement_percentage(self) -> float:
        """
        Calculate percentage improvement from initial to best solution

        # Function calculates subject improvement
        # Method quantifies predicate progress
        # Operation computes object efficiency

        Returns:
            Percentage improvement
        """
        if self.initial_best_fitness == 0:
            return 0.0

        improvement = self.best_fitness - self.initial_best_fitness
        return (improvement / abs(self.initial_best_fitness)) * 100

    def plot_history(
        self, show_diversity: bool = True, filename: Optional[str] = None
    ) -> Any:
        """
        Plot fitness and diversity history

        # Function plots subject history
        # Method visualizes predicate convergence
        # Operation displays object progress

        Args:
            show_diversity: Whether to show diversity on second axis
            filename: If provided, save plot to this file

        Returns:
            Matplotlib figure object
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot fitness history
        ax1.plot(self.fitness_history, "b-", label="Best Fitness")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Fitness", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Add diversity on second axis if requested
        if show_diversity and self.diversity_history:
            ax2 = ax1.twinx()
            ax2.plot(self.diversity_history, "r-", label="Swarm Diversity")
            ax2.set_ylabel("Diversity", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

        # Add title and grid
        plt.title("Particle Swarm Optimization")
        ax1.grid(True, alpha=0.3)

        # Add text box with statistics
        stats_text = (
            f"Initial Fitness: {self.initial_best_fitness:.4f}\n"
            f"Final Fitness: {self.best_fitness:.4f}\n"
            f"Improvement: {self.improvement_percentage():.2f}%\n"
            f"Iterations: {self.iterations}\n"
            f"Swarm Size: {self.swarm_size}\n"
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
        if show_diversity and self.diversity_history:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        plt.tight_layout()

        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        return fig


class TopologyType(Enum):
    """
    Swarm topology types for PSO

    # Class defines subject topologies
    # Enumeration specifies predicate options
    # Type catalogs object variants
    """

    GLOBAL = "global"  # Global best topology
    RING = "ring"  # Ring topology
    VON_NEUMANN = "von_neumann"  # Von Neumann topology (grid)
    RANDOM = "random"  # Random topology
    FOCAL = "focal"  # Focal topology (star)


@dataclass
class PSOConfig:
    """
    Configuration for particle swarm optimization

    # Class configures subject algorithm
    # Structure defines predicate parameters
    # Object specifies PSO settings
    """

    swarm_size: int = 50
    max_iterations: int = 100
    inertia_weight: float = 0.7
    cognitive_coefficient: float = 1.5  # Personal best weight (c1)
    social_coefficient: float = 1.5  # Global best weight (c2)
    max_velocity: Optional[float] = (
        None  # Maximum velocity (fraction of domain range)
    )
    topology: TopologyType = TopologyType.GLOBAL
    neighborhood_size: int = 3  # For ring and random topologies
    dynamic_inertia: bool = True  # Whether to decrease inertia over time
    final_inertia_weight: float = 0.4  # Final inertia weight when dynamic
    max_time_seconds: Optional[float] = None
    random_seed: Optional[int] = None
    early_stopping_iterations: Optional[int] = None  # Stop if no improvement
    early_stopping_threshold: Optional[float] = None  # Improvement threshold
    parallel: bool = False
    max_workers: Optional[int] = None


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization (PSO) implementation

    # Class optimizes subject problems
    # Optimizer executes predicate search
    # Engine finds object solutions
    """

    def __init__(self, config: Optional[PSOConfig] = None):
        """
        Initialize PSO optimizer

        # Function initializes subject optimizer
        # Method prepares predicate framework
        # Operation configures object parameters

        Args:
            config: Configuration for the algorithm
        """
        self.config = config or PSOConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

        # Initialize neighborhood topology
        self.neighborhood = None

        logger.info(
            f"Particle Swarm Optimization initialized with seed: {self.config.random_seed}"
        )

    def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: List[Tuple[float, float]],
        maximize: bool = True,
    ) -> PSOResult:
        """
        Run particle swarm optimization

        # Function optimizes subject problem
        # Method executes predicate search
        # Operation finds object solution

        Args:
            fitness_function: Function to evaluate fitness (higher is better if maximize=True)
            dimensions: Number of dimensions in the search space
            bounds: List of (min, max) bounds for each dimension
            maximize: Whether to maximize (True) or minimize (False) the fitness function

        Returns:
            PSOResult containing optimization results
        """
        start_time = time.time()

        # Validate bounds
        if len(bounds) != dimensions:
            raise ValueError(
                f"Bounds length ({len(bounds)}) must match dimensions ({dimensions})"
            )

        # Create fitness function wrapper based on maximize/minimize
        if maximize:
            evaluate = fitness_function
        else:
            # Negate fitness for minimization
            evaluate = lambda x: -fitness_function(x)

        # Initialize swarm
        swarm = self._initialize_swarm(dimensions, bounds, evaluate)

        # Initialize neighborhood topology
        self._initialize_topology()

        # Find initial best position and fitness
        global_best_position = None
        global_best_fitness = float("-inf")

        for particle in swarm:
            if particle.best_fitness > global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position.copy()

        initial_best_fitness = global_best_fitness

        # Initialize history
        fitness_history = [global_best_fitness]
        diversity_history = [self._calculate_diversity(swarm)]

        logger.info(
            f"Starting PSO with initial best fitness: {global_best_fitness}"
        )

        # Run optimization with parallel or sequential approach
        if self.config.parallel:
            result = self._optimize_parallel(
                swarm,
                global_best_position,
                global_best_fitness,
                initial_best_fitness,
                evaluate,
                bounds,
                start_time,
            )
        else:
            # Main optimization loop
            iteration = 0
            iterations_without_improvement = 0

            while True:
                # Check stopping conditions
                if iteration >= self.config.max_iterations:
                    logger.info(f"Reached maximum iterations: {iteration}")
                    break

                if (
                    self.config.max_time_seconds
                    and time.time() - start_time > self.config.max_time_seconds
                ):
                    logger.info(
                        f"Reached maximum time: {self.config.max_time_seconds} seconds"
                    )
                    break

                if (
                    self.config.early_stopping_iterations
                    and iterations_without_improvement
                    >= self.config.early_stopping_iterations
                ):
                    logger.info(
                        f"Early stopping after {iterations_without_improvement} iterations without improvement"
                    )
                    break

                # Get inertia weight for this iteration
                inertia = self._get_inertia_weight(iteration)

                # Update velocities and positions
                for i, particle in enumerate(swarm):
                    # Get neighborhood best
                    neighborhood_best = self._get_neighborhood_best(swarm, i)

                    # Update velocity
                    cognitive_component = (
                        self.config.cognitive_coefficient
                        * self.rng.random()
                        * (particle.best_position - particle.position)
                    )

                    social_component = (
                        self.config.social_coefficient
                        * self.rng.random()
                        * (neighborhood_best - particle.position)
                    )

                    particle.velocity = (
                        inertia * particle.velocity
                        + cognitive_component
                        + social_component
                    )

                    # Apply velocity clamping if configured
                    if self.config.max_velocity is not None:
                        for d in range(dimensions):
                            bound_range = bounds[d][1] - bounds[d][0]
                            max_vel = bound_range * self.config.max_velocity
                            particle.velocity[d] = np.clip(
                                particle.velocity[d], -max_vel, max_vel
                            )

                    # Update position
                    particle.position += particle.velocity

                    # Apply bounds
                    for d in range(dimensions):
                        particle.position[d] = np.clip(
                            particle.position[d], bounds[d][0], bounds[d][1]
                        )

                    # Evaluate new position
                    particle.current_fitness = evaluate(particle.position)

                    # Update personal best
                    if particle.current_fitness > particle.best_fitness:
                        particle.best_fitness = particle.current_fitness
                        particle.best_position = particle.position.copy()

                    # Update global best
                    if particle.best_fitness > global_best_fitness:
                        global_best_fitness = particle.best_fitness
                        global_best_position = particle.best_position.copy()
                        iterations_without_improvement = 0

                # Check if there was an improvement
                if global_best_fitness <= fitness_history[-1]:
                    iterations_without_improvement += 1

                # Track history
                fitness_history.append(global_best_fitness)
                diversity_history.append(self._calculate_diversity(swarm))

                # Log progress
                if iteration % max(1, self.config.max_iterations // 10) == 0:
                    logger.info(
                        f"Iteration {iteration}/{self.config.max_iterations}, "
                        f"Best Fitness: {global_best_fitness:.6f}, "
                        f"Diversity: {diversity_history[-1]:.4f}"
                    )

                iteration += 1

            # Create result object
            result = PSOResult(
                best_position=global_best_position,
                best_fitness=(
                    global_best_fitness if maximize else -global_best_fitness
                ),
                initial_best_fitness=(
                    initial_best_fitness if maximize else -initial_best_fitness
                ),
                iterations=iteration,
                swarm_size=self.config.swarm_size,
                fitness_history=(
                    fitness_history
                    if maximize
                    else [-f for f in fitness_history]
                ),
                diversity_history=diversity_history,
                execution_time=time.time() - start_time,
                metadata={
                    "random_seed": self.config.random_seed,
                    "topology": self.config.topology.value,
                    "iterations_without_improvement": iterations_without_improvement,
                },
            )

        # Adjust result for minimization if needed
        if not maximize and hasattr(result, "fitness_history"):
            result.fitness_history = [-f for f in result.fitness_history]

        logger.info(
            f"Optimization complete: {result.best_fitness:.6f} "
            f"(improvement: {result.improvement_percentage():.2f}%)"
        )
        logger.info(f"Execution time: {result.execution_time:.2f}s")

        return result

    def _optimize_parallel(
        self,
        initial_swarm: List[Particle],
        initial_best_position: np.ndarray,
        initial_best_fitness: float,
        initial_global_best: float,
        evaluate: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        start_time: float,
    ) -> PSOResult:
        """
        Run PSO with parallel fitness evaluation

        # Function optimizes subject problem
        # Method executes predicate parallel
        # Operation finds object solution

        Args:
            initial_swarm: Initial particle swarm
            initial_best_position: Initial best position
            initial_best_fitness: Initial best fitness
            initial_global_best: Initial global best fitness
            evaluate: Fitness evaluation function
            bounds: Parameter bounds
            start_time: Start time of the algorithm

        Returns:
            PSOResult containing optimization results
        """
        # Copy initial state
        swarm = initial_swarm.copy()
        global_best_position = initial_best_position.copy()
        global_best_fitness = initial_best_fitness
        dimensions = len(bounds)

        # Initialize history
        fitness_history = [global_best_fitness]
        diversity_history = [self._calculate_diversity(swarm)]

        # Define worker function for parallel particle evaluation
        def evaluate_particles(particles, inertia, iteration, worker_id):
            results = []

            for i, particle in enumerate(particles):
                # Get neighborhood best (use global best for parallel implementation)
                neighborhood_best = global_best_position

                # Update velocity
                cognitive_component = (
                    self.config.cognitive_coefficient
                    * np.random.random()
                    * (particle.best_position - particle.position)
                )

                social_component = (
                    self.config.social_coefficient
                    * np.random.random()
                    * (neighborhood_best - particle.position)
                )

                new_velocity = (
                    inertia * particle.velocity
                    + cognitive_component
                    + social_component
                )

                # Apply velocity clamping if configured
                if self.config.max_velocity is not None:
                    for d in range(dimensions):
                        bound_range = bounds[d][1] - bounds[d][0]
                        max_vel = bound_range * self.config.max_velocity
                        new_velocity[d] = np.clip(
                            new_velocity[d], -max_vel, max_vel
                        )

                # Update position
                new_position = particle.position + new_velocity

                # Apply bounds
                for d in range(dimensions):
                    new_position[d] = np.clip(
                        new_position[d], bounds[d][0], bounds[d][1]
                    )

                # Evaluate new position
                fitness = evaluate(new_position)

                # Return updated particle
                results.append(
                    (
                        new_position,
                        new_velocity,
                        (
                            particle.best_position.copy()
                            if fitness <= particle.best_fitness
                            else new_position.copy()
                        ),
                        max(fitness, particle.best_fitness),
                        fitness,
                    )
                )

            return results

        # Main optimization loop
        iteration = 0
        iterations_without_improvement = 0

        while True:
            # Check stopping conditions
            if iteration >= self.config.max_iterations:
                logger.info(f"Reached maximum iterations: {iteration}")
                break

            if (
                self.config.max_time_seconds
                and time.time() - start_time > self.config.max_time_seconds
            ):
                logger.info(
                    f"Reached maximum time: {self.config.max_time_seconds} seconds"
                )
                break

            if (
                self.config.early_stopping_iterations
                and iterations_without_improvement
                >= self.config.early_stopping_iterations
            ):
                logger.info(
                    f"Early stopping after {iterations_without_improvement} iterations without improvement"
                )
                break

            # Get inertia weight for this iteration
            inertia = self._get_inertia_weight(iteration)

            # Divide swarm among workers
            worker_count = self.config.max_workers or 1
            chunk_size = max(1, len(swarm) // worker_count)
            chunks = []

            for i in range(0, len(swarm), chunk_size):
                chunks.append(swarm[i : i + chunk_size])

            # Use ProcessPoolExecutor for parallel evaluation
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                # Submit jobs
                futures = []
                for worker_id, chunk in enumerate(chunks):
                    future = executor.submit(
                        evaluate_particles, chunk, inertia, iteration, worker_id
                    )
                    futures.append(future)

                # Collect results
                all_results = []
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        all_results.extend(chunk_results)
                    except Exception as e:
                        logger.error(f"Error in worker: {e}")

            # Process evaluation results
            swarm = []

            for pos, vel, best_pos, best_fit, curr_fit in all_results:
                particle = Particle(
                    position=pos,
                    velocity=vel,
                    best_position=best_pos,
                    best_fitness=best_fit,
                    current_fitness=curr_fit,
                )
                swarm.append(particle)

                # Update global best
                if particle.best_fitness > global_best_fitness:
                    global_best_fitness = particle.best_fitness
                    global_best_position = particle.best_position.copy()
                    iterations_without_improvement = 0

            # Check if there was an improvement
            if global_best_fitness <= fitness_history[-1]:
                iterations_without_improvement += 1

            # Track history
            fitness_history.append(global_best_fitness)
            diversity_history.append(self._calculate_diversity(swarm))

            # Log progress
            if iteration % max(1, self.config.max_iterations // 10) == 0:
                logger.info(
                    f"Iteration {iteration}/{self.config.max_iterations}, "
                    f"Best Fitness: {global_best_fitness:.6f}, "
                    f"Diversity: {diversity_history[-1]:.4f}"
                )

            iteration += 1

        # Create result object
        return PSOResult(
            best_position=global_best_position,
            best_fitness=global_best_fitness,
            initial_best_fitness=initial_global_best,
            iterations=iteration,
            swarm_size=self.config.swarm_size,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            execution_time=time.time() - start_time,
            metadata={
                "random_seed": self.config.random_seed,
                "topology": self.config.topology.value,
                "parallel": True,
                "iterations_without_improvement": iterations_without_improvement,
            },
        )

    def _initialize_swarm(
        self,
        dimensions: int,
        bounds: List[Tuple[float, float]],
        evaluate: Callable[[np.ndarray], float],
    ) -> List[Particle]:
        """
        Initialize swarm of particles

        # Function initializes subject swarm
        # Method creates predicate particles
        # Operation generates object positions

        Args:
            dimensions: Number of dimensions
            bounds: Parameter bounds
            evaluate: Fitness evaluation function

        Returns:
            List of initialized particles
        """
        swarm = []

        for _ in range(self.config.swarm_size):
            # Initialize position within bounds
            position = np.zeros(dimensions)
            for d in range(dimensions):
                position[d] = self.rng.uniform(bounds[d][0], bounds[d][1])

            # Initialize velocity as a fraction of position range
            velocity = np.zeros(dimensions)
            for d in range(dimensions):
                bound_range = bounds[d][1] - bounds[d][0]
                velocity[d] = self.rng.uniform(
                    -0.1 * bound_range, 0.1 * bound_range
                )

            # Evaluate fitness
            fitness = evaluate(position)

            # Create particle
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=fitness,
                current_fitness=fitness,
            )

            swarm.append(particle)

        return swarm

    def _initialize_topology(self) -> None:
        """
        Initialize neighborhood topology

        # Function initializes subject topology
        # Method creates predicate neighborhoods
        # Operation defines object connections
        """
        swarm_size = self.config.swarm_size
        topology = self.config.topology

        if topology == TopologyType.GLOBAL:
            # Global topology - all particles are connected to all others
            # No explicit neighborhood structure needed
            self.neighborhood = None

        elif topology == TopologyType.RING:
            # Ring topology - each particle connected to k neighbors
            k = min(self.config.neighborhood_size, swarm_size - 1)
            self.neighborhood = [[] for _ in range(swarm_size)]

            for i in range(swarm_size):
                # Add k/2 neighbors on each side of the ring
                for j in range(1, k // 2 + 1):
                    # Add previous neighbors (wrap around)
                    prev_idx = (i - j) % swarm_size
                    self.neighborhood[i].append(prev_idx)

                    # Add next neighbors (wrap around)
                    next_idx = (i + j) % swarm_size
                    self.neighborhood[i].append(next_idx)

        elif topology == TopologyType.VON_NEUMANN:
            # Von Neumann (grid) topology
            # Arrange particles in a 2D grid and connect to 4 neighbors
            self.neighborhood = [[] for _ in range(swarm_size)]

            # Find grid dimensions approximating a square
            grid_size = int(np.ceil(np.sqrt(swarm_size)))

            for i in range(swarm_size):
                # Calculate 2D grid coordinates
                row = i // grid_size
                col = i % grid_size

                # Add the 4 von Neumann neighbors (N, E, S, W)
                neighbors = [
                    ((row - 1) % grid_size) * grid_size + col,  # North
                    row * grid_size + ((col + 1) % grid_size),  # East
                    ((row + 1) % grid_size) * grid_size + col,  # South
                    row * grid_size + ((col - 1) % grid_size),  # West
                ]

                # Only add valid neighbors
                for n in neighbors:
                    if 0 <= n < swarm_size:
                        self.neighborhood[i].append(n)

        elif topology == TopologyType.RANDOM:
            # Random topology - connect each particle to k random others
            k = min(self.config.neighborhood_size, swarm_size - 1)
            self.neighborhood = [[] for _ in range(swarm_size)]

            for i in range(swarm_size):
                # Select k random neighbors (excluding self)
                potential_neighbors = list(range(swarm_size))
                potential_neighbors.remove(i)
                if len(potential_neighbors) > k:
                    neighbors = self.rng.choice(
                        potential_neighbors, k, replace=False
                    )
                else:
                    neighbors = potential_neighbors

                self.neighborhood[i] = list(neighbors)

        elif topology == TopologyType.FOCAL:
            # Focal (star) topology - one central particle connected to all others
            self.neighborhood = [[] for _ in range(swarm_size)]

            # Particle 0 is the focal point, connected to all others
            self.neighborhood[0] = list(range(1, swarm_size))

            # All other particles connected only to the focal point
            for i in range(1, swarm_size):
                self.neighborhood[i] = [0]

    def _get_neighborhood_best(
        self, swarm: List[Particle], particle_idx: int
    ) -> np.ndarray:
        """
        Get best position in the particle's neighborhood

        # Function finds subject neighborhood
        # Method identifies predicate best
        # Operation locates object optimum

        Args:
            swarm: List of particles
            particle_idx: Index of the current particle

        Returns:
            Best position in the neighborhood
        """
        topology = self.config.topology

        if topology == TopologyType.GLOBAL or self.neighborhood is None:
            # Return global best
            best_fitness = float("-inf")
            best_pos = None

            for particle in swarm:
                if particle.best_fitness > best_fitness:
                    best_fitness = particle.best_fitness
                    best_pos = particle.best_position

            return best_pos

        # Get neighborhood particles
        neighbors = self.neighborhood[particle_idx] + [
            particle_idx
        ]  # Include self

        # Find best in neighborhood
        best_fitness = float("-inf")
        best_pos = None

        for idx in neighbors:
            particle = swarm[idx]
            if particle.best_fitness > best_fitness:
                best_fitness = particle.best_fitness
                best_pos = particle.best_position

        return best_pos

    def _get_inertia_weight(self, iteration: int) -> float:
        """
        Get inertia weight for current iteration

        # Function calculates subject inertia
        # Method determines predicate weight
        # Operation computes object parameter

        Args:
            iteration: Current iteration

        Returns:
            Inertia weight
        """
        if not self.config.dynamic_inertia:
            return self.config.inertia_weight

        # Linear decrease from initial to final inertia
        weight_range = (
            self.config.inertia_weight - self.config.final_inertia_weight
        )
        progress = min(1.0, iteration / self.config.max_iterations)

        return self.config.inertia_weight - weight_range * progress

    def _calculate_diversity(self, swarm: List[Particle]) -> float:
        """
        Calculate diversity of the swarm

        # Function calculates subject diversity
        # Method measures predicate spread
        # Operation quantifies object variance

        Args:
            swarm: List of particles

        Returns:
            Diversity measure
        """
        if len(swarm) <= 1:
            return 0.0

        # Calculate centroid of the swarm
        positions = np.array([p.position for p in swarm])
        centroid = np.mean(positions, axis=0)

        # Calculate average distance to centroid
        distances = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
        avg_distance = np.mean(distances)

        return avg_distance


# Example usage
def example_objective_function(x: np.ndarray) -> float:
    """
    Example objective function (sphere function) - minimize

    # Function evaluates subject objective
    # Method calculates predicate value
    # Operation computes object fitness

    Args:
        x: Position in n-dimensional space

    Returns:
        Objective value (lower is better)
    """
    return -np.sum(x**2)  # Negative for maximization
