"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MONTE-CARLO-SIMULATION-0001         │
// │ 📁 domain       : Mathematics, Monte Carlo                  │
// │ 🧠 description  : Monte Carlo simulation implementation     │
// │                  for stochastic modeling                    │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_MONTE_CARLO                         │
// │ 🧩 dependencies : numpy, pandas, scipy                     │
// │ 🔧 tool_usage   : Simulation, Prediction                   │
// │ 📡 input_type   : Probability distributions, parameters     │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : simulation, prediction                    │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Monte Carlo Simulation Module
--------------------------
Provides a comprehensive Monte Carlo simulation framework
for stochastic modeling and prediction in intelligence analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import time
import matplotlib.pyplot as plt
from scipy import stats

# Function configures subject logger
# Method initializes predicate output
# Operation sets object format
logger = logging.getLogger("monte_carlo.simulation")
logger.setLevel(logging.INFO)


@dataclass
class SimulationResult:
    """
    Data class for storing Monte Carlo simulation results

    # Class stores subject results
    # Container holds predicate outputs
    # Structure formats object data
    """

    name: str
    iterations: int
    raw_data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def mean(self) -> float:
        """
        Calculate mean of simulation results

        # Function calculates subject mean
        # Method computes predicate average
        # Operation determines object center

        Returns:
            Mean value
        """
        return np.mean(self.raw_data)

    @property
    def median(self) -> float:
        """
        Calculate median of simulation results

        # Function calculates subject median
        # Method computes predicate middle
        # Operation determines object center

        Returns:
            Median value
        """
        return np.median(self.raw_data)

    @property
    def std_dev(self) -> float:
        """
        Calculate standard deviation of simulation results

        # Function calculates subject deviation
        # Method computes predicate spread
        # Operation determines object variance

        Returns:
            Standard deviation
        """
        return np.std(self.raw_data)

    @property
    def percentiles(self) -> Dict[int, float]:
        """
        Calculate key percentiles of simulation results

        # Function calculates subject percentiles
        # Method computes predicate quantiles
        # Operation determines object distribution

        Returns:
            Dictionary of percentile values (5, 25, 50, 75, 95)
        """
        return {
            5: np.percentile(self.raw_data, 5),
            25: np.percentile(self.raw_data, 25),
            50: np.percentile(self.raw_data, 50),
            75: np.percentile(self.raw_data, 75),
            95: np.percentile(self.raw_data, 95),
        }

    @property
    def confidence_interval(
        self, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for simulation results

        # Function calculates subject interval
        # Method computes predicate bounds
        # Operation determines object confidence

        Args:
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = self.mean
        std_err = stats.sem(self.raw_data)
        interval = stats.t.interval(
            confidence, len(self.raw_data) - 1, loc=mean, scale=std_err
        )
        return interval

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert simulation results to pandas DataFrame

        # Function converts subject results
        # Method transforms predicate data
        # Operation formats object frame

        Returns:
            DataFrame with simulation results
        """
        df = pd.DataFrame(
            {"iteration": range(len(self.raw_data)), "value": self.raw_data}
        )

        # Add metadata columns
        for key, value in self.metadata.items():
            df[key] = value

        return df

    def plot_histogram(
        self,
        bins: int = 30,
        show_stats: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot histogram of simulation results

        # Function plots subject histogram
        # Method visualizes predicate distribution
        # Operation displays object frequency

        Args:
            bins: Number of histogram bins
            show_stats: Whether to show statistics on plot
            filename: If provided, save plot to this file

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        ax.hist(
            self.raw_data,
            bins=bins,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )

        # Add vertical lines for key statistics
        ax.axvline(
            self.mean,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Mean: {self.mean:.4f}",
        )
        ax.axvline(
            self.median,
            color="green",
            linestyle="-.",
            alpha=0.8,
            label=f"Median: {self.median:.4f}",
        )

        if show_stats:
            # Add percentile lines
            percentiles = self.percentiles
            ax.axvline(
                percentiles[5],
                color="purple",
                linestyle=":",
                alpha=0.6,
                label=f"5th %: {percentiles[5]:.4f}",
            )
            ax.axvline(
                percentiles[95],
                color="purple",
                linestyle=":",
                alpha=0.6,
                label=f"95th %: {percentiles[95]:.4f}",
            )

            # Add text box with statistics
            stats_text = (
                f"Mean: {self.mean:.4f}\n"
                f"Median: {self.median:.4f}\n"
                f"Std Dev: {self.std_dev:.4f}\n"
                f"5th %: {percentiles[5]:.4f}\n"
                f"95th %: {percentiles[95]:.4f}\n"
                f"Iterations: {self.iterations}"
            )

            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Add title and labels
        ax.set_title(f"Monte Carlo Simulation Results: {self.name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        return fig


class MCSimulation:
    """
    Monte Carlo simulation framework

    # Class simulates subject models
    # Simulator runs predicate trials
    # Engine processes object stochasticity
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulation framework

        # Function initializes subject simulator
        # Method prepares predicate engine
        # Operation configures object randomness

        Args:
            random_seed: Seed for random number generator
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        logger.info(
            f"Monte Carlo simulation initialized with seed: {random_seed}"
        )

    def run_simulation(
        self,
        simulation_function: Callable[..., Any],
        iterations: int = 10000,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        name: str = "Simulation",
        **kwargs,
    ) -> SimulationResult:
        """
        Run a Monte Carlo simulation

        # Function runs subject simulation
        # Method executes predicate trials
        # Operation performs object calculations

        Args:
            simulation_function: Function that performs a single simulation iteration
            iterations: Number of simulation iterations
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers
            name: Name of the simulation
            **kwargs: Additional arguments to pass to simulation function

        Returns:
            SimulationResult object with simulation results
        """
        start_time = time.time()
        logger.info(
            f"Starting Monte Carlo simulation '{name}' with {iterations} iterations"
        )

        if parallel:
            results = self._run_parallel(
                simulation_function, iterations, max_workers, **kwargs
            )
        else:
            results = self._run_sequential(
                simulation_function, iterations, **kwargs
            )

        execution_time = time.time() - start_time
        logger.info(
            f"Completed Monte Carlo simulation in {execution_time:.2f} seconds"
        )

        return SimulationResult(
            name=name,
            iterations=iterations,
            raw_data=np.array(results),
            metadata={
                "execution_time": execution_time,
                "parallel": parallel,
                "max_workers": max_workers,
                "random_seed": self.random_seed,
            },
        )

    def _run_sequential(
        self, simulation_function: Callable[..., Any], iterations: int, **kwargs
    ) -> List[Any]:
        """
        Run simulation sequentially

        # Function runs subject iterations
        # Method executes predicate sequence
        # Operation performs object trials

        Args:
            simulation_function: Function that performs a single simulation iteration
            iterations: Number of simulation iterations
            **kwargs: Additional arguments to pass to simulation function

        Returns:
            List of simulation results
        """
        results = []

        for i in range(iterations):
            # Pass random state to ensure proper seeding per iteration
            result = simulation_function(
                random_state=self.rng, iteration=i, **kwargs
            )
            results.append(result)

            # Log progress for long simulations
            if i % max(1, iterations // 10) == 0 and i > 0:
                logger.info(
                    f"Simulation progress: {i}/{iterations} iterations ({i/iterations*100:.1f}%)"
                )

        return results

    def _run_parallel(
        self,
        simulation_function: Callable[..., Any],
        iterations: int,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Run simulation in parallel

        # Function runs subject iterations
        # Method executes predicate parallel
        # Operation performs object trials

        Args:
            simulation_function: Function that performs a single simulation iteration
            iterations: Number of simulation iterations
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments to pass to simulation function

        Returns:
            List of simulation results
        """
        results = []

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for i in range(iterations):
                # Create a different seed for each process
                seed = (
                    None if self.random_seed is None else self.random_seed + i
                )
                future = executor.submit(
                    simulation_function,
                    random_state=np.random.RandomState(seed),
                    iteration=i,
                    **kwargs,
                )
                futures.append(future)

            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in simulation iteration {i}: {e}")

                # Log progress for long simulations
                if i % max(1, iterations // 10) == 0 and i > 0:
                    logger.info(
                        f"Simulation progress: {i}/{iterations} iterations ({i/iterations*100:.1f}%)"
                    )

        return results

    def sensitivity_analysis(
        self,
        simulation_function: Callable[..., Any],
        parameter_ranges: Dict[str, List[Any]],
        iterations_per_param: int = 1000,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Dict[Any, SimulationResult]]:
        """
        Perform sensitivity analysis by varying parameters

        # Function analyzes subject sensitivity
        # Method varies predicate parameters
        # Operation measures object response

        Args:
            simulation_function: Function that performs a single simulation iteration
            parameter_ranges: Dictionary mapping parameter names to lists of values
            iterations_per_param: Number of iterations for each parameter value
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers
            **kwargs: Additional fixed arguments to pass to simulation function

        Returns:
            Dictionary mapping parameter names to dictionaries of parameter values to SimulationResult objects
        """
        results = {}

        for param_name, param_values in parameter_ranges.items():
            logger.info(
                f"Running sensitivity analysis for parameter '{param_name}'"
            )
            param_results = {}

            for value in param_values:
                # Update kwargs with current parameter value
                current_kwargs = {**kwargs, param_name: value}

                # Run simulation with current parameter value
                simulation_name = f"{param_name}={value}"
                result = self.run_simulation(
                    simulation_function,
                    iterations=iterations_per_param,
                    parallel=parallel,
                    max_workers=max_workers,
                    name=simulation_name,
                    **current_kwargs,
                )

                param_results[value] = result
                logger.info(
                    f"Completed parameter value {param_name}={value}, mean: {result.mean:.4f}"
                )

            results[param_name] = param_results

        return results

    def plot_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Dict[Any, SimulationResult]],
        plot_type: str = "boxplot",
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot sensitivity analysis results

        # Function plots subject sensitivity
        # Method visualizes predicate analysis
        # Operation displays object comparison

        Args:
            sensitivity_results: Results from sensitivity_analysis method
            plot_type: Type of plot ('boxplot', 'violin', or 'line')
            filename: If provided, save plot to this file

        Returns:
            Matplotlib figure object
        """
        num_params = len(sensitivity_results)
        fig, axes = plt.subplots(
            num_params, 1, figsize=(12, 5 * num_params), squeeze=False
        )

        for i, (param_name, param_results) in enumerate(
            sensitivity_results.items()
        ):
            ax = axes[i, 0]

            # Extract parameter values and results
            param_values = list(param_results.keys())

            if plot_type == "boxplot":
                # Create boxplot
                data = [result.raw_data for result in param_results.values()]
                ax.boxplot(data, labels=[str(val) for val in param_values])
                ax.set_title(f"Sensitivity Analysis: {param_name}")
                ax.set_xlabel(param_name)
                ax.set_ylabel("Output Value")
                ax.grid(True, alpha=0.3)

            elif plot_type == "violin":
                # Create violin plot
                data = [result.raw_data for result in param_results.values()]
                ax.violinplot(data, showmeans=True, showmedians=True)
                ax.set_xticks(range(1, len(param_values) + 1))
                ax.set_xticklabels([str(val) for val in param_values])
                ax.set_title(f"Sensitivity Analysis: {param_name}")
                ax.set_xlabel(param_name)
                ax.set_ylabel("Output Value")
                ax.grid(True, alpha=0.3)

            elif plot_type == "line":
                # Create line plot with error bands
                means = [result.mean for result in param_results.values()]
                stds = [result.std_dev for result in param_results.values()]

                ax.plot(
                    param_values, means, "o-", color="steelblue", label="Mean"
                )
                ax.fill_between(
                    param_values,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color="steelblue",
                    alpha=0.2,
                    label="±1 Std Dev",
                )
                ax.set_title(f"Sensitivity Analysis: {param_name}")
                ax.set_xlabel(param_name)
                ax.set_ylabel("Output Value")
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()

        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def scenario_analysis(
        scenarios: Dict[str, Dict[str, Any]],
        simulation_function: Callable[..., Any],
        iterations: int = 10000,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, SimulationResult]:
        """
        Perform scenario analysis with different parameter sets

        # Function analyzes subject scenarios
        # Method compares predicate conditions
        # Operation evaluates object outcomes

        Args:
            scenarios: Dictionary mapping scenario names to parameter dictionaries
            simulation_function: Function that performs a single simulation iteration
            iterations: Number of iterations per scenario
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers
            random_seed: Seed for random number generator

        Returns:
            Dictionary mapping scenario names to SimulationResult objects
        """
        results = {}

        # Create simulation instance
        simulator = MCSimulation(random_seed=random_seed)

        for scenario_name, params in scenarios.items():
            logger.info(f"Running scenario analysis: {scenario_name}")

            # Run simulation with scenario parameters
            result = simulator.run_simulation(
                simulation_function,
                iterations=iterations,
                parallel=parallel,
                max_workers=max_workers,
                name=scenario_name,
                **params,
            )

            results[scenario_name] = result
            logger.info(
                f"Completed scenario: {scenario_name}, mean: {result.mean:.4f}"
            )

        return results


# Example usage
def example_simulation(
    random_state: np.random.RandomState,
    iteration: int,
    mean: float = 0.0,
    std_dev: float = 1.0,
) -> float:
    """
    Example simulation function that generates a random sample

    # Function simulates subject sample
    # Method generates predicate value
    # Operation produces object instance

    Args:
        random_state: Random state for reproducibility
        iteration: Current iteration number
        mean: Mean of normal distribution
        std_dev: Standard deviation of normal distribution

    Returns:
        Random sample from normal distribution
    """
    return random_state.normal(mean, std_dev)
