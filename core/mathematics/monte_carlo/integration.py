"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MONTE-CARLO-INTEGRATION-0001        │
// │ 📁 domain       : Mathematics, Numerical Integration        │
// │ 🧠 description  : Monte Carlo numerical integration          │
// │                  methods for multidimensional integrals     │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_MONTE_CARLO                         │
// │ 🧩 dependencies : numpy, scipy                             │
// │ 🔧 tool_usage   : Integration, Calculation                 │
// │ 📡 input_type   : Mathematical functions                   │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : integration, calculation                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Monte Carlo Integration Module
---------------------------
Provides numerical integration methods using Monte Carlo techniques,
particularly useful for high-dimensional integrals that are difficult
to solve with conventional quadrature methods.
"""

import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("monte_carlo.integration")
logger.setLevel(logging.INFO)


@dataclass
class IntegrationResult:
    """
    Data class for storing Monte Carlo integration results

    # Class stores subject results
    # Container holds predicate outputs
    # Structure formats object data
    """

    integral_value: float
    error_estimate: float
    iterations: int
    dimension: int
    method: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """
        String representation of integration result

        # Function formats subject result
        # Method displays predicate output
        # Operation presents object value

        Returns:
            Formatted string representation
        """
        return (
            f"Integral Value: {self.integral_value:.6e} ± {self.error_estimate:.6e}\n"
            f"Method: {self.method}\n"
            f"Iterations: {self.iterations}\n"
            f"Dimension: {self.dimension}\n"
            f"Execution Time: {self.execution_time:.3f} seconds"
        )


class MCIntegration:
    """
    Monte Carlo numerical integration methods

    # Class integrates subject functions
    # Integrator calculates predicate values
    # Engine approximates object integrals
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo integration engine

        # Function initializes subject integrator
        # Method prepares predicate calculator
        # Operation configures object randomness

        Args:
            random_seed: Seed for random number generator
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        logger.info(
            f"Monte Carlo integration initialized with seed: {random_seed}"
        )

    def integrate_uniform(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        iterations: int = 100000,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> IntegrationResult:
        """
        Integrate a function over a hypercube using uniform sampling

        # Function integrates subject function
        # Method calculates predicate integral
        # Operation approximates object value

        Args:
            func: Function to integrate (takes argument vector of same dimension as domain)
            domain: List of (min, max) tuples defining integration domain
            iterations: Number of Monte Carlo iterations
            parallel: Whether to use parallel computation
            max_workers: Maximum number of parallel workers

        Returns:
            IntegrationResult containing integral approximation and error estimate
        """
        start_time = time.time()
        dimension = len(domain)

        logger.info(
            f"Starting Monte Carlo integration in {dimension}D with {iterations} samples"
        )

        # Calculate volume of integration domain
        volume = 1.0
        for dim_min, dim_max in domain:
            volume *= dim_max - dim_min

        if parallel:
            integral_sum, integral_sum_squared = (
                self._integrate_uniform_parallel(
                    func, domain, iterations, max_workers
                )
            )
        else:
            integral_sum, integral_sum_squared = (
                self._integrate_uniform_sequential(func, domain, iterations)
            )

        # Calculate integral estimate and error
        integral_mean = integral_sum / iterations
        integral_value = volume * integral_mean

        # Calculate error estimate (standard error of the mean)
        variance = integral_sum_squared / iterations - integral_mean**2
        error_estimate = volume * np.sqrt(max(0, variance) / iterations)

        execution_time = time.time() - start_time

        logger.info(
            f"Integration complete: {integral_value:.6e} ± {error_estimate:.6e}"
        )
        logger.info(f"Execution time: {execution_time:.3f} seconds")

        return IntegrationResult(
            integral_value=integral_value,
            error_estimate=error_estimate,
            iterations=iterations,
            dimension=dimension,
            method="uniform",
            execution_time=execution_time,
            metadata={
                "volume": volume,
                "random_seed": self.random_seed,
                "parallel": parallel,
                "max_workers": max_workers,
            },
        )

    def _integrate_uniform_sequential(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        iterations: int,
    ) -> Tuple[float, float]:
        """
        Perform uniform Monte Carlo integration sequentially

        # Function performs subject integration
        # Method calculates predicate samples
        # Operation evaluates object points

        Args:
            func: Function to integrate
            domain: Integration domain
            iterations: Number of iterations

        Returns:
            Tuple of (sum of function values, sum of squared function values)
        """
        dimension = len(domain)
        integral_sum = 0.0
        integral_sum_squared = 0.0

        for i in range(iterations):
            # Generate random point in the domain
            point = np.zeros(dimension)
            for j in range(dimension):
                dim_min, dim_max = domain[j]
                point[j] = self.rng.uniform(dim_min, dim_max)

            # Evaluate function at the point
            try:
                value = func(*point)
                integral_sum += value
                integral_sum_squared += value**2
            except Exception as e:
                logger.warning(f"Function evaluation failed at {point}: {e}")

            # Log progress for long integrations
            if i % max(1, iterations // 10) == 0 and i > 0:
                logger.info(
                    f"Integration progress: {i}/{iterations} iterations ({i/iterations*100:.1f}%)"
                )

        return integral_sum, integral_sum_squared

    def _integrate_uniform_parallel(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        iterations: int,
        max_workers: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Perform uniform Monte Carlo integration in parallel

        # Function performs subject integration
        # Method calculates predicate samples
        # Operation evaluates object points

        Args:
            func: Function to integrate
            domain: Integration domain
            iterations: Number of iterations
            max_workers: Maximum number of parallel workers

        Returns:
            Tuple of (sum of function values, sum of squared function values)
        """

        # Define a worker function to evaluate batch of points
        def worker_func(batch_size, worker_seed):
            # Create a separate random state for this worker
            worker_rng = np.random.RandomState(worker_seed)
            dimension = len(domain)
            local_sum = 0.0
            local_sum_squared = 0.0

            for _ in range(batch_size):
                # Generate random point in the domain
                point = np.zeros(dimension)
                for j in range(dimension):
                    dim_min, dim_max = domain[j]
                    point[j] = worker_rng.uniform(dim_min, dim_max)

                # Evaluate function at the point
                try:
                    value = func(*point)
                    local_sum += value
                    local_sum_squared += value**2
                except Exception as e:
                    # Log error but continue processing
                    pass

            return (local_sum, local_sum_squared)

        # Determine batch size per worker
        worker_count = max_workers or 1
        batch_size = max(1, iterations // (worker_count * 4))
        num_batches = (
            iterations + batch_size - 1
        ) // batch_size  # Ceiling division

        integral_sum = 0.0
        integral_sum_squared = 0.0

        # Use ProcessPoolExecutor for parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = []
            for i in range(num_batches):
                # Use different seed for each batch
                batch_seed = (
                    None if self.random_seed is None else self.random_seed + i
                )

                # Adjust batch size for the last batch
                current_batch_size = min(
                    batch_size, iterations - i * batch_size
                )
                if current_batch_size <= 0:
                    break

                future = executor.submit(
                    worker_func, current_batch_size, batch_seed
                )
                futures.append(future)

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    local_sum, local_sum_squared = future.result()
                    integral_sum += local_sum
                    integral_sum_squared += local_sum_squared
                    completed += 1

                    # Log progress
                    if completed % max(1, num_batches // 10) == 0:
                        logger.info(
                            f"Integration progress: {completed}/{num_batches} batches ({completed/num_batches*100:.1f}%)"
                        )

                except Exception as e:
                    logger.error(f"Error in worker batch: {e}")

        return integral_sum, integral_sum_squared

    def integrate_importance(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        pdf: Callable[..., float],
        sampler: Callable[[np.random.RandomState, int], np.ndarray],
        iterations: int = 100000,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> IntegrationResult:
        """
        Integrate a function using importance sampling

        # Function integrates subject function
        # Method applies predicate importance
        # Operation samples object efficiently

        Args:
            func: Function to integrate
            domain: List of (min, max) tuples defining integration domain
            pdf: Probability density function used for sampling
            sampler: Function that generates samples from the pdf
            iterations: Number of Monte Carlo iterations
            parallel: Whether to use parallel computation
            max_workers: Maximum number of parallel workers

        Returns:
            IntegrationResult containing integral approximation and error estimate
        """
        start_time = time.time()
        dimension = len(domain)

        logger.info(
            f"Starting importance sampling integration in {dimension}D with {iterations} samples"
        )

        if parallel:
            integral_sum, integral_sum_squared = (
                self._integrate_importance_parallel(
                    func, pdf, sampler, iterations, dimension, max_workers
                )
            )
        else:
            integral_sum, integral_sum_squared = (
                self._integrate_importance_sequential(
                    func, pdf, sampler, iterations, dimension
                )
            )

        # Calculate integral estimate and error
        integral_value = integral_sum / iterations

        # Calculate error estimate
        variance = integral_sum_squared / iterations - integral_value**2
        error_estimate = np.sqrt(max(0, variance) / iterations)

        execution_time = time.time() - start_time

        logger.info(
            f"Integration complete: {integral_value:.6e} ± {error_estimate:.6e}"
        )
        logger.info(f"Execution time: {execution_time:.3f} seconds")

        return IntegrationResult(
            integral_value=integral_value,
            error_estimate=error_estimate,
            iterations=iterations,
            dimension=dimension,
            method="importance",
            execution_time=execution_time,
            metadata={
                "random_seed": self.random_seed,
                "parallel": parallel,
                "max_workers": max_workers,
            },
        )

    def _integrate_importance_sequential(
        self,
        func: Callable[..., float],
        pdf: Callable[..., float],
        sampler: Callable[[np.random.RandomState, int], np.ndarray],
        iterations: int,
        dimension: int,
    ) -> Tuple[float, float]:
        """
        Perform importance sampling integration sequentially

        # Function performs subject integration
        # Method calculates predicate samples
        # Operation evaluates object points

        Args:
            func: Function to integrate
            pdf: Probability density function
            sampler: Function to sample from pdf
            iterations: Number of iterations
            dimension: Dimension of the integration

        Returns:
            Tuple of (sum of weighted function values, sum of squared weighted function values)
        """
        integral_sum = 0.0
        integral_sum_squared = 0.0

        for i in range(iterations):
            # Generate sample point using provided sampler
            point = sampler(self.rng, dimension)

            # Evaluate function and pdf at the point
            try:
                f_value = func(*point)
                p_value = pdf(*point)

                if p_value > 0:
                    weight = f_value / p_value
                    integral_sum += weight
                    integral_sum_squared += weight**2
            except Exception as e:
                logger.warning(f"Function evaluation failed at {point}: {e}")

            # Log progress for long integrations
            if i % max(1, iterations // 10) == 0 and i > 0:
                logger.info(
                    f"Integration progress: {i}/{iterations} iterations ({i/iterations*100:.1f}%)"
                )

        return integral_sum, integral_sum_squared

    def _integrate_importance_parallel(
        self,
        func: Callable[..., float],
        pdf: Callable[..., float],
        sampler: Callable[[np.random.RandomState, int], np.ndarray],
        iterations: int,
        dimension: int,
        max_workers: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Perform importance sampling integration in parallel

        # Function performs subject integration
        # Method calculates predicate samples
        # Operation evaluates object points

        Args:
            func: Function to integrate
            pdf: Probability density function
            sampler: Function to sample from pdf
            iterations: Number of iterations
            dimension: Dimension of the integration
            max_workers: Maximum number of parallel workers

        Returns:
            Tuple of (sum of weighted function values, sum of squared weighted function values)
        """

        # Define a worker function to evaluate batch of points
        def worker_func(batch_size, worker_seed):
            # Create a separate random state for this worker
            worker_rng = np.random.RandomState(worker_seed)
            local_sum = 0.0
            local_sum_squared = 0.0

            for _ in range(batch_size):
                # Generate sample point using provided sampler
                point = sampler(worker_rng, dimension)

                # Evaluate function and pdf at the point
                try:
                    f_value = func(*point)
                    p_value = pdf(*point)

                    if p_value > 0:
                        weight = f_value / p_value
                        local_sum += weight
                        local_sum_squared += weight**2
                except Exception as e:
                    # Log error but continue processing
                    pass

            return (local_sum, local_sum_squared)

        # Determine batch size per worker
        worker_count = max_workers or 1
        batch_size = max(1, iterations // (worker_count * 4))
        num_batches = (
            iterations + batch_size - 1
        ) // batch_size  # Ceiling division

        integral_sum = 0.0
        integral_sum_squared = 0.0

        # Use ProcessPoolExecutor for parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = []
            for i in range(num_batches):
                # Use different seed for each batch
                batch_seed = (
                    None if self.random_seed is None else self.random_seed + i
                )

                # Adjust batch size for the last batch
                current_batch_size = min(
                    batch_size, iterations - i * batch_size
                )
                if current_batch_size <= 0:
                    break

                future = executor.submit(
                    worker_func, current_batch_size, batch_seed
                )
                futures.append(future)

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    local_sum, local_sum_squared = future.result()
                    integral_sum += local_sum
                    integral_sum_squared += local_sum_squared
                    completed += 1

                    # Log progress
                    if completed % max(1, num_batches // 10) == 0:
                        logger.info(
                            f"Integration progress: {completed}/{num_batches} batches ({completed/num_batches*100:.1f}%)"
                        )

                except Exception as e:
                    logger.error(f"Error in worker batch: {e}")

        return integral_sum, integral_sum_squared

    def stratified_sampling(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        strata_per_dim: int = 5,
        samples_per_stratum: int = 5,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> IntegrationResult:
        """
        Integrate a function using stratified sampling

        # Function integrates subject function
        # Method applies predicate stratification
        # Operation partitions object domain

        Args:
            func: Function to integrate
            domain: List of (min, max) tuples defining integration domain
            strata_per_dim: Number of strata per dimension
            samples_per_stratum: Number of samples per stratum
            parallel: Whether to use parallel computation
            max_workers: Maximum number of parallel workers

        Returns:
            IntegrationResult containing integral approximation and error estimate
        """
        start_time = time.time()
        dimension = len(domain)

        # Total number of strata is (strata_per_dim)^dimension
        total_strata = strata_per_dim**dimension
        total_samples = total_strata * samples_per_stratum

        logger.info(f"Starting stratified sampling integration in {dimension}D")
        logger.info(
            f"Using {strata_per_dim} strata per dimension ({total_strata} total strata)"
        )
        logger.info(
            f"Using {samples_per_stratum} samples per stratum ({total_samples} total samples)"
        )

        # Calculate volume of integration domain
        volume = 1.0
        for dim_min, dim_max in domain:
            volume *= dim_max - dim_min

        # Volume of each stratum
        stratum_volume = volume / total_strata

        # Generate all strata indices
        strata_indices = self._generate_strata_indices(
            dimension, strata_per_dim
        )

        if parallel:
            stratum_results = self._stratified_sampling_parallel(
                func,
                domain,
                strata_indices,
                strata_per_dim,
                samples_per_stratum,
                max_workers,
            )
        else:
            stratum_results = self._stratified_sampling_sequential(
                func,
                domain,
                strata_indices,
                strata_per_dim,
                samples_per_stratum,
            )

        # Aggregate results across all strata
        stratum_means = np.array([result[0] for result in stratum_results])
        stratum_variances = np.array([result[1] for result in stratum_results])

        # Calculate integral estimate
        integral_value = stratum_volume * np.sum(stratum_means)

        # Calculate error estimate
        # For stratified sampling, we use the variance of the stratum means
        error_estimate = stratum_volume * np.sqrt(
            np.sum(stratum_variances) / (samples_per_stratum * total_strata)
        )

        execution_time = time.time() - start_time

        logger.info(
            f"Integration complete: {integral_value:.6e} ± {error_estimate:.6e}"
        )
        logger.info(f"Execution time: {execution_time:.3f} seconds")

        return IntegrationResult(
            integral_value=integral_value,
            error_estimate=error_estimate,
            iterations=total_samples,
            dimension=dimension,
            method="stratified",
            execution_time=execution_time,
            metadata={
                "strata_per_dim": strata_per_dim,
                "total_strata": total_strata,
                "samples_per_stratum": samples_per_stratum,
                "volume": volume,
                "stratum_volume": stratum_volume,
                "random_seed": self.random_seed,
                "parallel": parallel,
                "max_workers": max_workers,
            },
        )

    def _generate_strata_indices(
        self, dimension: int, strata_per_dim: int
    ) -> List[List[int]]:
        """
        Generate all possible strata indices

        # Function generates subject indices
        # Method creates predicate combinations
        # Operation forms object partitions

        Args:
            dimension: Dimension of the integration domain
            strata_per_dim: Number of strata per dimension

        Returns:
            List of strata indices, where each index is a list of dimension values
        """
        if dimension == 1:
            return [[i] for i in range(strata_per_dim)]

        # Recursive generation for higher dimensions
        lower_dim_indices = self._generate_strata_indices(
            dimension - 1, strata_per_dim
        )
        indices = []

        for i in range(strata_per_dim):
            for lower_index in lower_dim_indices:
                indices.append([i] + lower_index)

        return indices

    def _get_stratum_bounds(
        self,
        stratum_index: List[int],
        domain: List[Tuple[float, float]],
        strata_per_dim: int,
    ) -> List[Tuple[float, float]]:
        """
        Get bounds for a specific stratum

        # Function calculates subject bounds
        # Method determines predicate limits
        # Operation computes object extents

        Args:
            stratum_index: Index of the stratum in each dimension
            domain: Overall integration domain
            strata_per_dim: Number of strata per dimension

        Returns:
            List of (min, max) tuples for the stratum
        """
        stratum_bounds = []

        for dim, (dim_min, dim_max) in enumerate(domain):
            idx = stratum_index[dim]

            # Calculate bounds for this dimension
            width = (dim_max - dim_min) / strata_per_dim
            lower = dim_min + idx * width
            upper = dim_min + (idx + 1) * width

            stratum_bounds.append((lower, upper))

        return stratum_bounds

    def _stratified_sampling_sequential(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        strata_indices: List[List[int]],
        strata_per_dim: int,
        samples_per_stratum: int,
    ) -> List[Tuple[float, float]]:
        """
        Perform stratified sampling sequentially

        # Function performs subject sampling
        # Method evaluates predicate strata
        # Operation calculates object estimates

        Args:
            func: Function to integrate
            domain: Integration domain
            strata_indices: List of all strata indices
            strata_per_dim: Number of strata per dimension
            samples_per_stratum: Number of samples per stratum

        Returns:
            List of (mean, variance) tuples for each stratum
        """
        stratum_results = []
        dimension = len(domain)

        for i, stratum_index in enumerate(strata_indices):
            # Get bounds for this stratum
            stratum_bounds = self._get_stratum_bounds(
                stratum_index, domain, strata_per_dim
            )

            # Sample points within this stratum
            stratum_values = []

            for _ in range(samples_per_stratum):
                # Generate random point in the stratum
                point = np.zeros(dimension)
                for j in range(dimension):
                    dim_min, dim_max = stratum_bounds[j]
                    point[j] = self.rng.uniform(dim_min, dim_max)

                # Evaluate function at the point
                try:
                    value = func(*point)
                    stratum_values.append(value)
                except Exception as e:
                    logger.warning(
                        f"Function evaluation failed at {point}: {e}"
                    )

            # Calculate mean and variance for this stratum
            if stratum_values:
                stratum_mean = np.mean(stratum_values)
                stratum_variance = (
                    np.var(stratum_values, ddof=1) / len(stratum_values)
                    if len(stratum_values) > 1
                    else 0
                )
                stratum_results.append((stratum_mean, stratum_variance))
            else:
                # No valid samples in this stratum
                stratum_results.append((0.0, 0.0))

            # Log progress
            if (i + 1) % max(1, len(strata_indices) // 10) == 0:
                progress = (i + 1) / len(strata_indices) * 100
                logger.info(
                    f"Stratified sampling progress: {i+1}/{len(strata_indices)} strata ({progress:.1f}%)"
                )

        return stratum_results

    def _stratified_sampling_parallel(
        self,
        func: Callable[..., float],
        domain: List[Tuple[float, float]],
        strata_indices: List[List[int]],
        strata_per_dim: int,
        samples_per_stratum: int,
        max_workers: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """
        Perform stratified sampling in parallel

        # Function performs subject sampling
        # Method evaluates predicate strata
        # Operation calculates object estimates

        Args:
            func: Function to integrate
            domain: Integration domain
            strata_indices: List of all strata indices
            strata_per_dim: Number of strata per dimension
            samples_per_stratum: Number of samples per stratum
            max_workers: Maximum number of parallel workers

        Returns:
            List of (mean, variance) tuples for each stratum
        """

        # Define a worker function to evaluate a batch of strata
        def worker_func(strata_batch, worker_seed):
            # Create a separate random state for this worker
            worker_rng = np.random.RandomState(worker_seed)
            dimension = len(domain)
            batch_results = []

            for stratum_index in strata_batch:
                # Get bounds for this stratum
                stratum_bounds = []
                for dim, (dim_min, dim_max) in enumerate(domain):
                    idx = stratum_index[dim]
                    width = (dim_max - dim_min) / strata_per_dim
                    lower = dim_min + idx * width
                    upper = dim_min + (idx + 1) * width
                    stratum_bounds.append((lower, upper))

                # Sample points within this stratum
                stratum_values = []

                for _ in range(samples_per_stratum):
                    # Generate random point in the stratum
                    point = np.zeros(dimension)
                    for j in range(dimension):
                        dim_min, dim_max = stratum_bounds[j]
                        point[j] = worker_rng.uniform(dim_min, dim_max)

                    # Evaluate function at the point
                    try:
                        value = func(*point)
                        stratum_values.append(value)
                    except Exception:
                        # Continue with other samples
                        pass

                # Calculate mean and variance for this stratum
                if stratum_values:
                    stratum_mean = np.mean(stratum_values)
                    stratum_variance = (
                        np.var(stratum_values, ddof=1) / len(stratum_values)
                        if len(stratum_values) > 1
                        else 0
                    )
                    batch_results.append((stratum_mean, stratum_variance))
                else:
                    batch_results.append((0.0, 0.0))

            return batch_results

        # Determine batch size per worker
        worker_count = max_workers or 1
        batch_size = max(1, len(strata_indices) // (worker_count * 2))

        # Divide strata among workers
        strata_batches = []
        for i in range(0, len(strata_indices), batch_size):
            strata_batches.append(strata_indices[i : i + batch_size])

        all_results = []

        # Use ProcessPoolExecutor for parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = []
            for i, batch in enumerate(strata_batches):
                # Use different seed for each batch
                batch_seed = (
                    None if self.random_seed is None else self.random_seed + i
                )
                future = executor.submit(worker_func, batch, batch_seed)
                futures.append(future)

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed += 1

                    # Log progress
                    if completed % max(1, len(strata_batches) // 10) == 0:
                        progress = completed / len(strata_batches) * 100
                        logger.info(
                            f"Stratified sampling progress: {completed}/{len(strata_batches)} batches ({progress:.1f}%)"
                        )

                except Exception as e:
                    logger.error(f"Error in worker batch: {e}")

        return all_results


# Example usage
def example_integrand(x, y):
    """
    Example function to integrate: f(x,y) = x^2 + y^2

    # Function calculates subject value
    # Method evaluates predicate point
    # Operation computes object result

    Args:
        x: x-coordinate
        y: y-coordinate

    Returns:
        Function value at (x,y)
    """
    return x**2 + y**2
