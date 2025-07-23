"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-GENETIC-ALGORITHM-0001              │
// │ 📁 domain       : Mathematics, Optimization                 │
// │ 🧠 description  : Genetic algorithm implementation for      │
// │                  evolutionary optimization                  │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_OPTIMIZATION                        │
// │ 🧩 dependencies : numpy, time, logging                     │
// │ 🔧 tool_usage   : Optimization, Evolution                  │
// │ 📡 input_type   : Fitness functions, genome operations     │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : optimization, evolution                   │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Genetic Algorithm Module
---------------------
Implementation of genetic algorithms for evolutionary optimization
of complex problems by simulating natural selection processes.
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
    Protocol,
)
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("optimization.genetic_algorithm")
logger.setLevel(logging.INFO)

# Generic type for genome
T = TypeVar("T")


class SelectionMethod(Enum):
    """
    Selection methods for genetic algorithms

    # Class defines subject methods
    # Enumeration specifies predicate options
    # Type catalogs object variants
    """

    TOURNAMENT = "tournament"  # Tournament selection
    ROULETTE = "roulette"  # Roulette wheel selection
    RANK = "rank"  # Rank-based selection
    TRUNCATION = "truncation"  # Truncation selection
    CUSTOM = "custom"  # User-defined selection function


class CrossoverMethod(Enum):
    """
    Crossover methods for genetic algorithms

    # Class defines subject methods
    # Enumeration specifies predicate options
    # Type catalogs object variants
    """

    SINGLE_POINT = "single_point"  # Single-point crossover
    TWO_POINT = "two_point"  # Two-point crossover
    UNIFORM = "uniform"  # Uniform crossover
    CUSTOM = "custom"  # User-defined crossover function


@dataclass
class EvolutionResult(Generic[T]):
    """
    Result of genetic algorithm optimization

    # Class stores subject results
    # Container holds predicate data
    # Structure formats object output
    """

    best_genome: T
    best_fitness: float
    initial_best_fitness: float
    generations: int
    population_size: int
    fitness_history: List[float]
    diversity_history: List[float]
    execution_time: float
    selection_method: SelectionMethod
    crossover_method: CrossoverMethod
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

    def plot_fitness_history(
        self, show_diversity: bool = True, filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot fitness history during evolution

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
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Add diversity on second axis if requested
        if show_diversity and self.diversity_history:
            ax2 = ax1.twinx()
            ax2.plot(self.diversity_history, "r-", label="Population Diversity")
            ax2.set_ylabel("Diversity", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

        # Add title and grid
        plt.title(
            f"Genetic Algorithm Evolution - {self.selection_method.value} selection"
        )
        ax1.grid(True, alpha=0.3)

        # Add text box with statistics
        stats_text = (
            f"Initial Fitness: {self.initial_best_fitness:.4f}\n"
            f"Final Fitness: {self.best_fitness:.4f}\n"
            f"Improvement: {self.improvement_percentage():.2f}%\n"
            f"Generations: {self.generations}\n"
            f"Population Size: {self.population_size}\n"
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


@dataclass
class GeneticConfig:
    """
    Configuration for genetic algorithm

    # Class configures subject algorithm
    # Structure defines predicate parameters
    # Object specifies evolution settings
    """

    population_size: int = 100
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism_count: int = 2
    tournament_size: int = 5  # For tournament selection
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    custom_selection_function: Optional[
        Callable[[List[Tuple[T, float]], int, np.random.RandomState], List[T]]
    ] = None
    custom_crossover_function: Optional[
        Callable[[T, T, np.random.RandomState], Tuple[T, T]]
    ] = None
    max_time_seconds: Optional[float] = None
    random_seed: Optional[int] = None
    early_stopping_generations: Optional[int] = (
        None  # Stop if no improvement for this many generations
    )
    early_stopping_threshold: Optional[float] = (
        None  # Minimum improvement to reset early stopping counter
    )
    parallel: bool = False
    max_workers: Optional[int] = None


class GenomeOperations(Protocol[T]):
    """
    Protocol defining genome operations for genetic algorithm

    # Class defines subject interface
    # Protocol specifies predicate contract
    # Object declares operation methods
    """

    def create_genome(self, random_state: np.random.RandomState) -> T:
        """
        Create a random genome

        # Function creates subject genome
        # Method generates predicate individual
        # Operation initializes object chromosome

        Args:
            random_state: Random state for reproducibility

        Returns:
            Randomly generated genome
        """
        ...

    def calculate_fitness(self, genome: T) -> float:
        """
        Calculate fitness of a genome

        # Function calculates subject fitness
        # Method evaluates predicate quality
        # Operation measures object performance

        Args:
            genome: Genome to evaluate

        Returns:
            Fitness value (higher is better)
        """
        ...

    def mutate(
        self,
        genome: T,
        mutation_rate: float,
        random_state: np.random.RandomState,
    ) -> T:
        """
        Mutate a genome

        # Function mutates subject genome
        # Method modifies predicate individual
        # Operation changes object genes

        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutation
            random_state: Random state for reproducibility

        Returns:
            Mutated genome
        """
        ...

    def crossover(
        self, parent1: T, parent2: T, random_state: np.random.RandomState
    ) -> Tuple[T, T]:
        """
        Perform crossover between two parents

        # Function crosses subject genomes
        # Method combines predicate parents
        # Operation produces object children

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            random_state: Random state for reproducibility

        Returns:
            Tuple of (child1, child2)
        """
        ...

    def calculate_diversity(self, population: List[T]) -> float:
        """
        Calculate diversity of the population

        # Function calculates subject diversity
        # Method measures predicate variance
        # Operation quantifies object differences

        Args:
            population: List of genomes

        Returns:
            Diversity measure (higher is more diverse)
        """
        ...


class GeneticAlgorithm(Generic[T]):
    """
    Genetic algorithm optimizer

    # Class optimizes subject problems
    # Evolution executes predicate search
    # Engine finds object solutions
    """

    def __init__(self, config: Optional[GeneticConfig] = None):
        """
        Initialize genetic algorithm optimizer

        # Function initializes subject optimizer
        # Method prepares predicate framework
        # Operation configures object parameters

        Args:
            config: Configuration for the algorithm
        """
        self.config = config or GeneticConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        logger.info(
            f"Genetic algorithm initialized with seed: {self.config.random_seed}"
        )

    def evolve(self, genome_ops: GenomeOperations[T]) -> EvolutionResult[T]:
        """
        Run genetic algorithm evolution

        # Function evolves subject population
        # Method executes predicate search
        # Operation finds object solution

        Args:
            genome_ops: Genome operations

        Returns:
            EvolutionResult containing optimization results
        """
        start_time = time.time()

        # Create initial population
        population = []
        for _ in range(self.config.population_size):
            genome = genome_ops.create_genome(self.rng)
            population.append(genome)

        # Evaluate initial population
        fitness_values = []
        for genome in population:
            fitness = genome_ops.calculate_fitness(genome)
            fitness_values.append(fitness)

        # Track best individual and history
        best_genome = population[np.argmax(fitness_values)]
        best_fitness = max(fitness_values)
        initial_best_fitness = best_fitness

        fitness_history = [best_fitness]
        diversity_history = [genome_ops.calculate_diversity(population)]

        logger.info(
            f"Starting evolution with initial best fitness: {best_fitness}"
        )

        # Run evolution with parallel or sequential approach
        if self.config.parallel:
            result = self._evolve_parallel(
                population, fitness_values, genome_ops, start_time
            )
        else:
            # Evolution loop
            generation = 0
            generations_without_improvement = 0

            while True:
                # Check stopping conditions
                if generation >= self.config.max_generations:
                    logger.info(f"Reached maximum generations: {generation}")
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
                    self.config.early_stopping_generations
                    and generations_without_improvement
                    >= self.config.early_stopping_generations
                ):
                    logger.info(
                        f"Early stopping after {generations_without_improvement} generations without improvement"
                    )
                    break

                # Create next generation
                next_population = []

                # Elitism - keep best individuals
                if self.config.elitism_count > 0:
                    elite_indices = np.argsort(fitness_values)[
                        -self.config.elitism_count :
                    ]
                    for idx in elite_indices:
                        next_population.append(population[idx])

                # Fill the rest of the population with offspring
                while len(next_population) < self.config.population_size:
                    # Select parents
                    parent1 = self._select_individual(
                        population, fitness_values
                    )
                    parent2 = self._select_individual(
                        population, fitness_values
                    )

                    # Apply crossover
                    if self.rng.random() < self.config.crossover_rate:
                        child1, child2 = self._crossover(
                            parent1, parent2, genome_ops
                        )
                    else:
                        # No crossover, just copy parents
                        child1, child2 = parent1, parent2

                    # Apply mutation
                    child1 = genome_ops.mutate(
                        child1, self.config.mutation_rate, self.rng
                    )
                    child2 = genome_ops.mutate(
                        child2, self.config.mutation_rate, self.rng
                    )

                    # Add to next generation
                    next_population.append(child1)
                    if len(next_population) < self.config.population_size:
                        next_population.append(child2)

                # Update population
                population = next_population

                # Evaluate new population
                fitness_values = []
                for genome in population:
                    fitness = genome_ops.calculate_fitness(genome)
                    fitness_values.append(fitness)

                # Update best individual
                current_best_fitness = max(fitness_values)
                current_best_genome = population[np.argmax(fitness_values)]

                # Check for improvement
                if current_best_fitness > best_fitness:
                    # Calculate improvement
                    improvement = current_best_fitness - best_fitness
                    relative_improvement = (
                        improvement / abs(best_fitness)
                        if best_fitness != 0
                        else float("inf")
                    )

                    # Update best
                    best_fitness = current_best_fitness
                    best_genome = current_best_genome

                    # Reset or increment early stopping counter
                    if (
                        self.config.early_stopping_threshold is None
                        or relative_improvement
                        > self.config.early_stopping_threshold
                    ):
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                else:
                    generations_without_improvement += 1

                # Track history
                fitness_history.append(best_fitness)
                diversity_history.append(
                    genome_ops.calculate_diversity(population)
                )

                # Log progress
                if generation % max(1, self.config.max_generations // 10) == 0:
                    logger.info(
                        f"Generation {generation}/{self.config.max_generations}, "
                        f"Best Fitness: {best_fitness:.6f}, "
                        f"Diversity: {diversity_history[-1]:.4f}"
                    )

                generation += 1

            # Create result object
            result = EvolutionResult(
                best_genome=best_genome,
                best_fitness=best_fitness,
                initial_best_fitness=initial_best_fitness,
                generations=generation,
                population_size=self.config.population_size,
                fitness_history=fitness_history,
                diversity_history=diversity_history,
                execution_time=time.time() - start_time,
                selection_method=self.config.selection_method,
                crossover_method=self.config.crossover_method,
                metadata={
                    "random_seed": self.config.random_seed,
                    "generations_without_improvement": generations_without_improvement,
                },
            )

        logger.info(
            f"Evolution complete: {result.best_fitness:.6f} (improvement: {result.improvement_percentage():.2f}%)"
        )
        logger.info(f"Execution time: {result.execution_time:.2f}s")

        return result

    def _evolve_parallel(
        self,
        initial_population: List[T],
        initial_fitness_values: List[float],
        genome_ops: GenomeOperations[T],
        start_time: float,
    ) -> EvolutionResult[T]:
        """
        Run genetic algorithm evolution with parallel fitness evaluation

        # Function evolves subject population
        # Method executes predicate parallel
        # Operation finds object solution

        Args:
            initial_population: Initial population of genomes
            initial_fitness_values: Initial fitness values
            genome_ops: Genome operations
            start_time: Start time of the algorithm

        Returns:
            EvolutionResult containing optimization results
        """
        # Copy initial state
        population = initial_population.copy()
        fitness_values = initial_fitness_values.copy()

        # Track best individual and history
        best_genome = population[np.argmax(fitness_values)]
        best_fitness = max(fitness_values)
        initial_best_fitness = best_fitness

        fitness_history = [best_fitness]
        diversity_history = [genome_ops.calculate_diversity(population)]

        logger.info(
            f"Starting parallel evolution with initial best fitness: {best_fitness}"
        )

        # Define worker function for parallel fitness evaluation
        def evaluate_genomes(genomes, worker_id):
            results = []
            for genome in genomes:
                fitness = genome_ops.calculate_fitness(genome)
                results.append((genome, fitness))
            return results

        # Evolution loop
        generation = 0
        generations_without_improvement = 0

        while True:
            # Check stopping conditions
            if generation >= self.config.max_generations:
                logger.info(f"Reached maximum generations: {generation}")
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
                self.config.early_stopping_generations
                and generations_without_improvement
                >= self.config.early_stopping_generations
            ):
                logger.info(
                    f"Early stopping after {generations_without_improvement} generations without improvement"
                )
                break

            # Create next generation
            next_population = []

            # Elitism - keep best individuals
            if self.config.elitism_count > 0:
                elite_indices = np.argsort(fitness_values)[
                    -self.config.elitism_count :
                ]
                for idx in elite_indices:
                    next_population.append(population[idx])

            # Fill the rest of the population with offspring
            while len(next_population) < self.config.population_size:
                # Select parents
                parent1 = self._select_individual(population, fitness_values)
                parent2 = self._select_individual(population, fitness_values)

                # Apply crossover
                if self.rng.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(
                        parent1, parent2, genome_ops
                    )
                else:
                    # No crossover, just copy parents
                    child1, child2 = parent1, parent2

                # Apply mutation
                child1 = genome_ops.mutate(
                    child1, self.config.mutation_rate, self.rng
                )
                child2 = genome_ops.mutate(
                    child2, self.config.mutation_rate, self.rng
                )

                # Add to next generation
                next_population.append(child1)
                if len(next_population) < self.config.population_size:
                    next_population.append(child2)

            # Update population
            population = next_population

            # Evaluate population in parallel
            new_fitness_values = []

            # Divide population among workers
            worker_count = self.config.max_workers or 1
            chunk_size = max(1, len(population) // worker_count)
            chunks = []

            for i in range(0, len(population), chunk_size):
                chunks.append(population[i : i + chunk_size])

            # Use ProcessPoolExecutor for parallel evaluation
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                # Submit jobs
                futures = []
                for worker_id, chunk in enumerate(chunks):
                    future = executor.submit(evaluate_genomes, chunk, worker_id)
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
            population = []
            fitness_values = []

            for genome, fitness in all_results:
                population.append(genome)
                fitness_values.append(fitness)

            # Update best individual
            if fitness_values:  # Check if we have valid results
                current_best_fitness = max(fitness_values)
                current_best_genome = population[np.argmax(fitness_values)]

                # Check for improvement
                if current_best_fitness > best_fitness:
                    # Calculate improvement
                    improvement = current_best_fitness - best_fitness
                    relative_improvement = (
                        improvement / abs(best_fitness)
                        if best_fitness != 0
                        else float("inf")
                    )

                    # Update best
                    best_fitness = current_best_fitness
                    best_genome = current_best_genome

                    # Reset or increment early stopping counter
                    if (
                        self.config.early_stopping_threshold is None
                        or relative_improvement
                        > self.config.early_stopping_threshold
                    ):
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                else:
                    generations_without_improvement += 1

            # Track history
            fitness_history.append(best_fitness)
            diversity_history.append(genome_ops.calculate_diversity(population))

            # Log progress
            if generation % max(1, self.config.max_generations // 10) == 0:
                logger.info(
                    f"Generation {generation}/{self.config.max_generations}, "
                    f"Best Fitness: {best_fitness:.6f}, "
                    f"Diversity: {diversity_history[-1]:.4f}"
                )

            generation += 1

        # Create result object
        return EvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            initial_best_fitness=initial_best_fitness,
            generations=generation,
            population_size=self.config.population_size,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            execution_time=time.time() - start_time,
            selection_method=self.config.selection_method,
            crossover_method=self.config.crossover_method,
            metadata={
                "random_seed": self.config.random_seed,
                "parallel": True,
                "generations_without_improvement": generations_without_improvement,
            },
        )

    def _select_individual(
        self, population: List[T], fitness_values: List[float]
    ) -> T:
        """
        Select an individual from the population using the configured selection method

        # Function selects subject individual
        # Method chooses predicate parent
        # Operation identifies object genome

        Args:
            population: Population of genomes
            fitness_values: Fitness values for each genome

        Returns:
            Selected genome
        """
        method = self.config.selection_method

        if method == SelectionMethod.TOURNAMENT:
            # Tournament selection
            tournament_size = min(self.config.tournament_size, len(population))
            tournament_indices = self.rng.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_idx]

        elif method == SelectionMethod.ROULETTE:
            # Roulette wheel selection
            fitness_sum = sum(fitness_values)
            if fitness_sum == 0:
                # If all fitness values are zero, select randomly
                return population[self.rng.randint(0, len(population))]

            # Normalize fitness values to create probabilities
            selection_probs = [f / fitness_sum for f in fitness_values]
            selected_idx = self.rng.choice(len(population), p=selection_probs)
            return population[selected_idx]

        elif method == SelectionMethod.RANK:
            # Rank-based selection
            # Sort indices by fitness (ascending)
            sorted_indices = np.argsort(fitness_values)
            # Calculate ranks (lowest fitness = rank 1)
            ranks = np.arange(1, len(sorted_indices) + 1)
            # Create probability distribution based on ranks
            rank_sum = sum(ranks)
            rank_probs = [r / rank_sum for r in ranks]
            # Select based on rank probability
            selected_rank = self.rng.choice(len(ranks), p=rank_probs)
            selected_idx = sorted_indices[selected_rank]
            return population[selected_idx]

        elif method == SelectionMethod.TRUNCATION:
            # Truncation selection - select from top 50%
            num_to_keep = max(1, len(population) // 2)
            top_indices = np.argsort(fitness_values)[-num_to_keep:]
            selected_idx = self.rng.choice(top_indices)
            return population[selected_idx]

        elif method == SelectionMethod.CUSTOM:
            # Custom selection function
            if self.config.custom_selection_function:
                # Create pairs of (genome, fitness)
                genome_fitness_pairs = list(zip(population, fitness_values))
                # Use custom function to select individuals
                selected = self.config.custom_selection_function(
                    genome_fitness_pairs, 1, self.rng
                )
                if selected:
                    return selected[0]
                else:
                    # Fallback to tournament selection if custom function fails
                    logger.warning(
                        "Custom selection function failed, falling back to tournament selection"
                    )
                    return self._select_individual(population, fitness_values)
            else:
                logger.warning(
                    "Custom selection method selected but no function provided. Using tournament."
                )
                return self._select_individual(population, fitness_values)

        else:
            # Default to tournament selection
            return self._select_individual(population, fitness_values)

    def _crossover(
        self, parent1: T, parent2: T, genome_ops: GenomeOperations[T]
    ) -> Tuple[T, T]:
        """
        Perform crossover between two parents using the configured crossover method

        # Function crosses subject genomes
        # Method combines predicate parents
        # Operation produces object children

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            genome_ops: Genome operations

        Returns:
            Tuple of (child1, child2)
        """
        method = self.config.crossover_method

        if method == CrossoverMethod.CUSTOM:
            # Custom crossover function
            if self.config.custom_crossover_function:
                return self.config.custom_crossover_function(
                    parent1, parent2, self.rng
                )
            else:
                logger.warning(
                    "Custom crossover method selected but no function provided. Using genome_ops.crossover."
                )
                return genome_ops.crossover(parent1, parent2, self.rng)

        # For all other methods, use the genome operations
        return genome_ops.crossover(parent1, parent2, self.rng)


# Example implementation of genome operations for binary strings
class BinaryGenomeOps:
    """
    Example implementation of genome operations for binary strings

    # Class implements subject operations
    # Implementation defines predicate methods
    # Object provides genome manipulations
    """

    def __init__(self, genome_length: int, target: Optional[str] = None):
        """
        Initialize binary genome operations

        # Function initializes subject operations
        # Method configures predicate parameters
        # Operation prepares object settings

        Args:
            genome_length: Length of binary strings
            target: Optional target binary string (for optimization)
        """
        self.genome_length = genome_length
        self.target = target

    def create_genome(self, random_state: np.random.RandomState) -> str:
        """
        Create a random binary string

        # Function creates subject genome
        # Method generates predicate string
        # Operation produces object binary

        Args:
            random_state: Random state for reproducibility

        Returns:
            Random binary string
        """
        return "".join(
            random_state.choice(["0", "1"]) for _ in range(self.genome_length)
        )

    def calculate_fitness(self, genome: str) -> float:
        """
        Calculate fitness - higher is better

        # Function calculates subject fitness
        # Method evaluates predicate quality
        # Operation measures object performance

        Args:
            genome: Binary string to evaluate

        Returns:
            Fitness value
        """
        if self.target:
            # Maximize matches with target
            matches = sum(1 for a, b in zip(genome, self.target) if a == b)
            return matches / self.genome_length
        else:
            # Example: maximize number of 1s
            return genome.count("1") / self.genome_length

    def mutate(
        self,
        genome: str,
        mutation_rate: float,
        random_state: np.random.RandomState,
    ) -> str:
        """
        Mutate a binary string by flipping bits

        # Function mutates subject genome
        # Method modifies predicate string
        # Operation flips object bits

        Args:
            genome: Binary string to mutate
            mutation_rate: Probability of mutation per bit
            random_state: Random state for reproducibility

        Returns:
            Mutated binary string
        """
        result = list(genome)
        for i in range(len(result)):
            if random_state.random() < mutation_rate:
                result[i] = "1" if result[i] == "0" else "0"
        return "".join(result)

    def crossover(
        self, parent1: str, parent2: str, random_state: np.random.RandomState
    ) -> Tuple[str, str]:
        """
        Perform single-point crossover

        # Function crosses subject genomes
        # Method combines predicate strings
        # Operation swaps object segments

        Args:
            parent1: First parent binary string
            parent2: Second parent binary string
            random_state: Random state for reproducibility

        Returns:
            Tuple of (child1, child2)
        """
        # Select crossover point
        crossover_point = random_state.randint(1, len(parent1) - 1)

        # Create children
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def calculate_diversity(self, population: List[str]) -> float:
        """
        Calculate diversity as average Hamming distance

        # Function calculates subject diversity
        # Method measures predicate variance
        # Operation quantifies object differences

        Args:
            population: List of binary strings

        Returns:
            Diversity measure
        """
        if len(population) <= 1:
            return 0.0

        total_distance = 0
        count = 0

        # Calculate average pairwise Hamming distance
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = sum(
                    1 for a, b in zip(population[i], population[j]) if a != b
                )
                total_distance += distance
                count += 1

        return (
            total_distance / (count * self.genome_length) if count > 0 else 0.0
        )
