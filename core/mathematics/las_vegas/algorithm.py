"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-LAS-VEGAS-ALGORITHM-0001            │
// │ 📁 domain       : Mathematics, Randomized Algorithms        │
// │ 🧠 description  : Las Vegas algorithm implementation for    │
// │                  randomized search with guaranteed results  │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_LAS_VEGAS                           │
// │ 🧩 dependencies : numpy, time, logging                     │
// │ 🔧 tool_usage   : Optimization, Search                     │
// │ 📡 input_type   : Problem specifications                   │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : optimization, problem-solving             │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Las Vegas Algorithm Module
-----------------------
Implementation of Las Vegas randomized algorithm framework that
always produces correct results but with probabilistic running time.
"""

import logging
import time
import numpy as np
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
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("las_vegas.algorithm")
logger.setLevel(logging.INFO)

# Generic type for solution
T = TypeVar("T")


@dataclass
class LasVegasConfig:
    """
    Configuration for Las Vegas algorithm

    # Class configures subject algorithm
    # Config defines predicate parameters
    # Object specifies algorithm settings
    """

    max_iterations: int = 1000
    max_time_seconds: Optional[float] = None
    random_seed: Optional[int] = None
    parallel: bool = False
    max_workers: Optional[int] = None
    verification_function: Optional[Callable[[Any], bool]] = None
    early_stopping_threshold: Optional[float] = None


@dataclass
class LasVegasSolution(Generic[T]):
    """
    Solution object for Las Vegas algorithm

    # Class stores subject solution
    # Container holds predicate result
    # Structure formats object data
    """

    solution: T
    iterations: int
    execution_time: float
    success: bool
    attempts: List[Tuple[bool, float]] = field(default_factory=list)

    @property
    def average_attempt_time(self) -> float:
        """
        Calculate average time per attempt

        # Function calculates subject average
        # Method computes predicate time
        # Operation determines object performance

        Returns:
            Average time per attempt in seconds
        """
        if not self.attempts:
            return 0.0
        return statistics.mean([t for _, t in self.attempts])

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate

        # Function calculates subject rate
        # Method computes predicate success
        # Operation determines object reliability

        Returns:
            Success rate as fraction between 0 and 1
        """
        if not self.attempts:
            return 0.0
        return sum(1 for success, _ in self.attempts if success) / len(
            self.attempts
        )


class LasVegasAlgorithm(Generic[T]):
    """
    Las Vegas algorithm implementation

    # Class implements subject algorithm
    # Engine executes predicate search
    # Framework performs object optimization
    """

    def __init__(self, config: Optional[LasVegasConfig] = None):
        """
        Initialize Las Vegas algorithm

        # Function initializes subject algorithm
        # Method prepares predicate framework
        # Operation configures object parameters

        Args:
            config: Configuration for the algorithm
        """
        self.config = config or LasVegasConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        logger.info(
            f"Las Vegas algorithm initialized with seed: {self.config.random_seed}"
        )

    def run(
        self,
        search_function: Callable[..., Optional[T]],
        verification_function: Optional[Callable[[T], bool]] = None,
        **kwargs,
    ) -> LasVegasSolution[T]:
        """
        Run Las Vegas algorithm

        # Function runs subject algorithm
        # Method executes predicate search
        # Operation finds object solution

        Args:
            search_function: Function that performs randomized search
            verification_function: Function to verify solution correctness
            **kwargs: Additional arguments to pass to search function

        Returns:
            LasVegasSolution object containing results
        """
        verify_fn = verification_function or self.config.verification_function
        start_time = time.time()
        iterations = 0
        attempts = []

        # Run search with parallel or sequential approach
        if self.config.parallel:
            solution, iterations, attempts = self._run_parallel(
                search_function, verify_fn, start_time, **kwargs
            )
        else:
            solution, iterations, attempts = self._run_sequential(
                search_function, verify_fn, start_time, **kwargs
            )

        execution_time = time.time() - start_time
        success = solution is not None

        if success:
            logger.info(
                f"Found solution in {iterations} iterations ({execution_time:.2f} seconds)"
            )
        else:
            logger.warning(
                f"Failed to find solution after {iterations} iterations ({execution_time:.2f} seconds)"
            )

        return LasVegasSolution(
            solution=solution,
            iterations=iterations,
            execution_time=execution_time,
            success=success,
            attempts=attempts,
        )

    def _run_sequential(
        self,
        search_function: Callable[..., Optional[T]],
        verification_function: Optional[Callable[[T], bool]],
        start_time: float,
        **kwargs,
    ) -> Tuple[Optional[T], int, List[Tuple[bool, float]]]:
        """
        Run Las Vegas algorithm sequentially

        # Function runs subject search
        # Method executes predicate iterations
        # Operation finds object solution

        Args:
            search_function: Function that performs randomized search
            verification_function: Function to verify solution correctness
            start_time: Start time of the algorithm
            **kwargs: Additional arguments to pass to search function

        Returns:
            Tuple of (solution, iterations, attempts)
        """
        solution = None
        iterations = 0
        attempts = []

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

            # Run a single search iteration
            iteration_start = time.time()
            try:
                # Pass random state to ensure proper seeding
                candidate_solution = search_function(
                    random_state=self.rng, iteration=iterations, **kwargs
                )

                # Check if solution was found
                if candidate_solution is not None:
                    # Verify solution if verification function is provided
                    if verification_function and not verification_function(
                        candidate_solution
                    ):
                        iterations += 1
                        attempts.append((False, time.time() - iteration_start))
                        continue

                    # Solution found and verified
                    solution = candidate_solution
                    iterations += 1
                    attempts.append((True, time.time() - iteration_start))
                    break

            except Exception as e:
                logger.error(f"Error in search iteration {iterations}: {e}")

            # No solution found in this iteration
            iterations += 1
            attempts.append((False, time.time() - iteration_start))

            # Log progress for long searches
            if iterations % max(1, self.config.max_iterations // 10) == 0:
                logger.info(
                    f"Search progress: {iterations}/{self.config.max_iterations} iterations "
                    f"({iterations/self.config.max_iterations*100:.1f}%)"
                )

        return solution, iterations, attempts

    def _run_parallel(
        self,
        search_function: Callable[..., Optional[T]],
        verification_function: Optional[Callable[[T], bool]],
        start_time: float,
        **kwargs,
    ) -> Tuple[Optional[T], int, List[Tuple[bool, float]]]:
        """
        Run Las Vegas algorithm in parallel

        # Function runs subject search
        # Method executes predicate parallel
        # Operation finds object solution

        Args:
            search_function: Function that performs randomized search
            verification_function: Function to verify solution correctness
            start_time: Start time of the algorithm
            **kwargs: Additional arguments to pass to search function

        Returns:
            Tuple of (solution, iterations, attempts)
        """

        # Define a worker function to run iterations
        def worker_func(batch_size, worker_seed, worker_start_iteration):
            # Create a separate random state for this worker
            worker_rng = np.random.RandomState(worker_seed)
            worker_attempts = []

            for i in range(batch_size):
                iteration = worker_start_iteration + i
                iteration_start = time.time()

                try:
                    # Run search with worker's random state
                    candidate = search_function(
                        random_state=worker_rng, iteration=iteration, **kwargs
                    )

                    # Check if solution was found
                    if candidate is not None:
                        # Verify solution if verification function is provided
                        if verification_function and not verification_function(
                            candidate
                        ):
                            worker_attempts.append(
                                (False, time.time() - iteration_start)
                            )
                            continue

                        # Solution found and verified
                        worker_attempts.append(
                            (True, time.time() - iteration_start)
                        )
                        return True, candidate, i + 1, worker_attempts

                except Exception as e:
                    # Log error but continue with other iterations
                    pass

                # No solution found in this iteration
                worker_attempts.append((False, time.time() - iteration_start))

                # Check time limit
                if (
                    self.config.max_time_seconds
                    and time.time() - start_time > self.config.max_time_seconds
                ):
                    return False, None, i + 1, worker_attempts

            # No solution found in this batch
            return False, None, batch_size, worker_attempts

        solution = None
        total_iterations = 0
        all_attempts = []

        # Determine batch size and number of batches
        worker_count = self.config.max_workers or 1
        batch_size = max(
            1, min(100, self.config.max_iterations // (worker_count * 2))
        )
        max_batches = (
            self.config.max_iterations + batch_size - 1
        ) // batch_size

        batch_number = 0

        # Use ProcessPoolExecutor for parallelism
        with ProcessPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            while batch_number < max_batches:
                # Check if we've exceeded max time
                if (
                    self.config.max_time_seconds
                    and time.time() - start_time > self.config.max_time_seconds
                ):
                    logger.info(
                        f"Reached maximum time: {self.config.max_time_seconds} seconds"
                    )
                    break

                # Check if we've found a solution
                if solution is not None:
                    break

                # Determine number of batches to run in this round
                remaining_iterations = (
                    self.config.max_iterations - total_iterations
                )
                remaining_batches = (
                    remaining_iterations + batch_size - 1
                ) // batch_size
                batches_this_round = min(worker_count * 2, remaining_batches)

                if batches_this_round <= 0:
                    break

                # Submit batches
                futures = []
                for i in range(batches_this_round):
                    current_batch = batch_number + i
                    worker_seed = (
                        None
                        if self.config.random_seed is None
                        else self.config.random_seed + current_batch
                    )
                    start_iteration = total_iterations + i * batch_size

                    # Adjust batch size for the last batch
                    current_batch_size = min(
                        batch_size, self.config.max_iterations - start_iteration
                    )
                    if current_batch_size <= 0:
                        break

                    future = executor.submit(
                        worker_func,
                        current_batch_size,
                        worker_seed,
                        start_iteration,
                    )
                    futures.append(future)

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        found, result, iterations_run, attempts = (
                            future.result()
                        )
                        all_attempts.extend(attempts)
                        total_iterations += iterations_run

                        if found:
                            solution = result
                            # Cancel remaining futures if possible
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break

                    except Exception as e:
                        logger.error(f"Error in worker batch: {e}")

                # Update batch number
                batch_number += batches_this_round

                # Log progress
                if batch_number % max(1, max_batches // 10) == 0:
                    logger.info(
                        f"Search progress: {total_iterations}/{self.config.max_iterations} iterations "
                        f"({total_iterations/self.config.max_iterations*100:.1f}%)"
                    )

        return solution, total_iterations, all_attempts

    def adaptive_restart(
        self,
        search_function: Callable[..., Optional[T]],
        verification_function: Optional[Callable[[T], bool]] = None,
        restart_factor: float = 2.0,
        initial_iterations: int = 10,
        max_restarts: int = 10,
        **kwargs,
    ) -> LasVegasSolution[T]:
        """
        Run Las Vegas algorithm with adaptive restarts

        # Function runs subject algorithm
        # Method implements predicate restart
        # Operation adapts object strategy

        Args:
            search_function: Function that performs randomized search
            verification_function: Function to verify solution correctness
            restart_factor: Factor to increase iterations by after each restart
            initial_iterations: Initial number of iterations before first restart
            max_restarts: Maximum number of restarts
            **kwargs: Additional arguments to pass to search function

        Returns:
            LasVegasSolution object containing results
        """
        start_time = time.time()
        verify_fn = verification_function or self.config.verification_function

        total_iterations = 0
        all_attempts = []

        for restart in range(max_restarts):
            # Calculate iterations for this run
            current_iterations = int(
                initial_iterations * (restart_factor**restart)
            )

            logger.info(
                f"Restart {restart+1}/{max_restarts}: Running with {current_iterations} iterations"
            )

            # Create temporary config with current iterations
            temp_config = LasVegasConfig(
                max_iterations=current_iterations,
                max_time_seconds=self.config.max_time_seconds,
                random_seed=(
                    None
                    if self.config.random_seed is None
                    else self.config.random_seed + restart
                ),
                parallel=self.config.parallel,
                max_workers=self.config.max_workers,
                verification_function=verify_fn,
                early_stopping_threshold=self.config.early_stopping_threshold,
            )

            # Run with current configuration
            temp_algorithm = LasVegasAlgorithm(temp_config)
            result = temp_algorithm.run(search_function, verify_fn, **kwargs)

            # Update totals
            total_iterations += result.iterations
            all_attempts.extend(result.attempts)

            # Check if solution was found
            if result.success:
                logger.info(
                    f"Solution found after {restart+1} restarts and {total_iterations} total iterations"
                )
                break

            # Check if we've exceeded max time
            if (
                self.config.max_time_seconds
                and time.time() - start_time > self.config.max_time_seconds
            ):
                logger.info(
                    f"Reached maximum time: {self.config.max_time_seconds} seconds"
                )
                break

        # Create final solution object
        execution_time = time.time() - start_time

        return LasVegasSolution(
            solution=result.solution if result.success else None,
            iterations=total_iterations,
            execution_time=execution_time,
            success=result.success if "result" in locals() else False,
            attempts=all_attempts,
        )


# Example usage
def example_search_function(
    random_state: np.random.RandomState,
    iteration: int,
    target: int = 42,
    range_max: int = 1000,
) -> Optional[int]:
    """
    Example search function that tries to find a specific number

    # Function searches subject number
    # Method samples predicate randomly
    # Operation finds object target

    Args:
        random_state: Random state for reproducibility
        iteration: Current iteration number
        target: Target number to find
        range_max: Maximum range for random sampling

    Returns:
        The target number if found, None otherwise
    """
    # Generate a random number
    guess = random_state.randint(0, range_max)

    # Return the target if found, None otherwise
    return guess if guess == target else None


def example_verification_function(solution: int) -> bool:
    """
    Example verification function

    # Function verifies subject solution
    # Method validates predicate result
    # Operation confirms object correctness

    Args:
        solution: Solution to verify

    Returns:
        True if solution is valid, False otherwise
    """
    # In this simple example, any non-None solution is valid
    return solution is not None
