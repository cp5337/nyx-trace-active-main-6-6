"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DEMO-OPTIMIZATION-0001              │
// │ 📁 domain       : Visualization, Optimization              │
// │ 🧠 description  : Advanced optimization algorithms          │
// │                  demonstration and visualization dashboard  │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_VISUALIZATION                       │
// │ 🧩 dependencies : streamlit, numpy, matplotlib             │
// │ 🔧 tool_usage   : Demonstration, Visualization             │
// │ 📡 input_type   : User interaction                         │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : visualization, demonstration              │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Advanced Optimization Algorithms Demonstration
-------------------------------------------
Interactive dashboard for visualizing and experimenting with
Monte Carlo, Las Vegas, Simulated Annealing, Genetic Algorithms,
and Particle Swarm Optimization techniques.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Callable, Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image
from datetime import datetime

# Import optimization algorithms
# Simulated Annealing
try:
    from core.mathematics.optimization.simulated_annealing import (
        SimulatedAnnealing, AnnealingConfig, CoolingSchedule, AnnealingResult
    )
    HAS_SIMULATED_ANNEALING = True
except ImportError:
    HAS_SIMULATED_ANNEALING = False

# Genetic Algorithm
try:
    from core.mathematics.optimization.genetic_algorithm import (
        GeneticAlgorithm, GeneticConfig, SelectionMethod, CrossoverMethod, 
        EvolutionResult, BinaryGenomeOps, GenomeOperations
    )
    HAS_GENETIC_ALGORITHM = True
except ImportError:
    HAS_GENETIC_ALGORITHM = False

# Particle Swarm Optimization
try:
    from core.mathematics.optimization.particle_swarm import (
        ParticleSwarmOptimization, PSOConfig, TopologyType, PSOResult
    )
    HAS_PARTICLE_SWARM = True
except ImportError:
    HAS_PARTICLE_SWARM = False

# Monte Carlo
try:
    from core.mathematics.monte_carlo.simulation import (
        MCSimulation, SimulationResult
    )
    from core.mathematics.monte_carlo.integration import (
        MCIntegration, IntegrationResult
    )
    HAS_MONTE_CARLO = True
except ImportError:
    HAS_MONTE_CARLO = False

# Las Vegas
try:
    from core.mathematics.las_vegas.algorithm import (
        LasVegasAlgorithm, LasVegasConfig, LasVegasSolution
    )
    HAS_LAS_VEGAS = True
except ImportError:
    HAS_LAS_VEGAS = False


# Function sets subject page
# Method configures predicate settings
# Operation establishes object layout
def page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="CTAS Advanced Optimization",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better spacing and readability
    st.markdown("""
    <style>
        .main {
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 0;
        }
        h1 {
            margin-top: 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
        }
        .stRadio [data-testid="stMarkdownContainer"] {
            display: flex;
        }
        .problem-card {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .simulation-results {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# Function initializes subject page
# Method prepares predicate content
# Operation displays object header
def initialize_page():
    """Initialize the dashboard layout and header section"""
    st.title("Advanced Optimization Algorithms")
    st.markdown("""
    This interactive dashboard allows you to explore and experiment with various 
    optimization and simulation algorithms used in the CTAS framework for 
    intelligence analysis and decision support.
    
    Select an algorithm from the sidebar to begin.
    """)


# --------------------------------
# Test Functions for Optimization
# --------------------------------

# Function defines subject problems
# Method specifies predicate options
# Operation catalogs object functions
def get_optimization_problems() -> Dict[str, Dict]:
    """Get dictionary of test optimization problems"""
    return {
        "Rosenbrock Function": {
            "function": rosenbrock_function,
            "bounds": [(-3, 3), (-3, 3)],
            "global_minimum": np.array([1.0, 1.0]),
            "global_minimum_value": 0.0,
            "description": """
            The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms.
            It has a narrow, curved valley which contains the global minimum. Finding the valley is relatively straightforward, 
            but converging to the global minimum is difficult.
            
            Defined as: f(x,y) = (1-x)² + 100(y-x²)²
            """
        },
        "Rastrigin Function": {
            "function": rastrigin_function,
            "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
            "global_minimum": np.array([0.0, 0.0]),
            "global_minimum_value": 0.0,
            "description": """
            The Rastrigin function is a highly multimodal test function. It has many local minima 
            arranged in a regular lattice, making it challenging for optimization algorithms to find the global minimum.
            
            Defined as: f(x₁, x₂, ..., xₙ) = 10n + ∑[xᵢ² - 10cos(2πxᵢ)]
            """
        },
        "Ackley Function": {
            "function": ackley_function,
            "bounds": [(-5, 5), (-5, 5)],
            "global_minimum": np.array([0.0, 0.0]),
            "global_minimum_value": 0.0,
            "description": """
            The Ackley function is characterized by a nearly flat outer region and a large hole at the center.
            It tests an algorithm's ability to escape local minima and find the global minimum.
            
            Defined as: f(x,y) = -20exp(-0.2√(0.5(x²+y²))) - exp(0.5(cos(2πx)+cos(2πy))) + e + 20
            """
        },
    }


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function (banana function)
    Global minimum at (1, 1) with value 0
    
    # Function calculates subject value
    # Method evaluates predicate position
    # Operation computes object fitness
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin function
    Global minimum at (0, 0) with value 0
    
    # Function calculates subject value
    # Method evaluates predicate position
    # Operation computes object fitness
    """
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)


def ackley_function(x: np.ndarray) -> float:
    """
    Ackley function
    Global minimum at (0, 0) with value 0
    
    # Function calculates subject value
    # Method evaluates predicate position
    # Operation computes object fitness
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    part1 = -a * np.exp(-b * np.sqrt(0.5 * sum(xi**2 for xi in x)))
    part2 = -np.exp(0.5 * sum(np.cos(c * xi) for xi in x))
    
    return part1 + part2 + a + np.exp(1)


def visualize_2d_function(
    function: Callable, 
    bounds: List[Tuple[float, float]], 
    points: List[np.ndarray] = None,
    title: str = "Objective Function",
    contour: bool = True
) -> go.Figure:
    """
    Create a 3D surface plot and optional contour plot of a 2D function
    
    # Function visualizes subject function
    # Method plots predicate surface
    # Operation displays object landscape
    
    Args:
        function: The function to visualize
        bounds: List of (min, max) bounds for each dimension
        points: Optional list of points to overlay on the plot
        title: Plot title
        contour: Whether to include a contour plot
        
    Returns:
        Plotly figure object
    """
    # Create a grid of points
    resolution = 100
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = function(np.array([X[i, j], Y[i, j]]))
    
    # Create subplots with 1 row and 2 columns if contour is True
    if contour:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'contour'}]],
            subplot_titles=["3D Surface", "Contour Map"],
            horizontal_spacing=0.05
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'surface'}]],
            subplot_titles=["3D Surface"]
        )
    
    # Add surface plot
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            contours={
                "z": {"show": True, "start": Z.min(), "end": Z.max(), "size": (Z.max()-Z.min())/10}
            }
        ),
        row=1, col=1
    )
    
    # Add contour plot if requested
    if contour:
        fig.add_trace(
            go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                contours={
                    "start": Z.min(),
                    "end": Z.max(),
                    "size": (Z.max()-Z.min())/20
                }
            ),
            row=1, col=2
        )
    
    # Add points if provided
    if points is not None:
        for i, point in enumerate(points):
            # Add point to 3D surface
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0]], y=[point[1]], z=[function(point)],
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name=f"Point {i+1}"
                ),
                row=1, col=1
            )
            
            # Add point to contour if present
            if contour:
                fig.add_trace(
                    go.Scatter(
                        x=[point[0]], y=[point[1]],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='x'),
                        name=f"Point {i+1}"
                    ),
                    row=1, col=2
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)",
            aspectratio=dict(x=1, y=1, z=0.8)
        )
    )
    
    if contour:
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="y", row=1, col=2)
    
    return fig


# --------------------------------
# Simulated Annealing
# --------------------------------

# Function implements subject page
# Method creates predicate UI
# Operation displays object interface
def simulated_annealing_page():
    """Simulated Annealing demonstration page"""
    st.header("Simulated Annealing Optimization")
    
    st.markdown("""
    Simulated Annealing is a probabilistic optimization algorithm inspired by the 
    physical process of annealing in metallurgy. It's designed to find global optima 
    in complex search spaces with many local minima.
    
    Key characteristics:
    - Accepts worse solutions with decreasing probability over time
    - Gradually transitions from exploration to exploitation
    - Effective for discrete and continuous optimization problems
    """)
    
    if not HAS_SIMULATED_ANNEALING:
        st.warning("Simulated Annealing module not found. Please ensure the core.mathematics.optimization package is properly installed.")
        return
    
    # Get optimization problems
    problems = get_optimization_problems()
    
    # Create two columns for settings and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Algorithm Settings")
        
        # Problem selection
        problem_name = st.selectbox(
            "Test Problem",
            list(problems.keys()),
            index=0
        )
        
        problem = problems[problem_name]
        
        with st.expander("Problem Description", expanded=False):
            st.markdown(problem["description"])
        
        # Algorithm parameters
        max_iterations = st.slider(
            "Maximum Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        initial_temp = st.slider(
            "Initial Temperature",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=10.0
        )
        
        final_temp = st.slider(
            "Final Temperature",
            min_value=0.0001,
            max_value=10.0,
            value=0.1,
            step=0.1,
            format="%.4f"
        )
        
        cooling_schedule = st.selectbox(
            "Cooling Schedule",
            options=[cs.value for cs in CoolingSchedule],
            index=0
        )
        
        cooling_factor = st.slider(
            "Cooling Factor (for exponential cooling)",
            min_value=0.7,
            max_value=0.999,
            value=0.95,
            step=0.01,
            format="%.3f"
        )
        
        # Run optimization
        if st.button("Run Optimization", key="run_sa"):
            # Show a spinner while running
            with st.spinner("Running Simulated Annealing..."):
                # Configure the algorithm
                config = AnnealingConfig(
                    initial_temperature=initial_temp,
                    final_temperature=final_temp,
                    cooling_schedule=CoolingSchedule(cooling_schedule),
                    cooling_factor=cooling_factor,
                    max_iterations=max_iterations,
                    random_seed=42
                )
                
                # Create optimizer
                optimizer = SimulatedAnnealing(config)
                
                # Create a wrapper for the objective function (SA minimizes)
                objective_fn = problem["function"]
                energy_fn = lambda x: objective_fn(x)
                
                # Define neighbor function
                def neighbor_fn(x, rng, temp):
                    # Scale perturbation based on temperature
                    scale = temp / initial_temp
                    
                    # Create copy of position
                    neighbor = x.copy()
                    
                    # Perturb each dimension
                    for i in range(len(neighbor)):
                        bound_range = problem["bounds"][i][1] - problem["bounds"][i][0]
                        neighbor[i] += rng.normal(0, 0.1 * bound_range * scale)
                        neighbor[i] = np.clip(neighbor[i], problem["bounds"][i][0], problem["bounds"][i][1])
                    
                    return neighbor
                
                # Random initial solution
                rng = np.random.RandomState(42)
                initial_solution = np.array([
                    rng.uniform(problem["bounds"][0][0], problem["bounds"][0][1]),
                    rng.uniform(problem["bounds"][1][0], problem["bounds"][1][1])
                ])
                
                # Run optimization
                result = optimizer.optimize(
                    initial_solution=initial_solution,
                    energy_function=energy_fn,
                    neighbor_function=neighbor_fn
                )
                
                # Store result in session state
                st.session_state.sa_result = result
                st.session_state.sa_problem = problem
                st.session_state.sa_initial_solution = initial_solution
    
    with col2:
        st.subheader("Visualization")
        
        # Visualize the problem
        if "sa_result" in st.session_state:
            result = st.session_state.sa_result
            problem = st.session_state.sa_problem
            initial_solution = st.session_state.sa_initial_solution
            
            # Get the objective function
            objective_fn = problem["function"]
            
            # Create function visualization
            points = [
                initial_solution,  # Initial solution
                result.best_solution,  # Final solution
                problem["global_minimum"]  # Global minimum
            ]
            
            fig = visualize_2d_function(
                objective_fn,
                problem["bounds"],
                points=points,
                title=f"{problem_name} Optimization"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show results
            st.markdown("### Optimization Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Initial Energy",
                    f"{result.initial_energy:.6f}",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Final Energy",
                    f"{result.best_energy:.6f}",
                    delta=f"{-abs(result.initial_energy - result.best_energy):.6f}",
                    delta_color="inverse"
                )
            
            with col_c:
                st.metric(
                    "Global Minimum",
                    f"{problem['global_minimum_value']:.6f}",
                    delta=f"{abs(result.best_energy - problem['global_minimum_value']):.6f}",
                    delta_color="inverse"
                )
            
            # Additional info
            col_d, col_e, col_f = st.columns(3)
            
            with col_d:
                st.metric(
                    "Iterations",
                    f"{result.iterations}",
                    delta=None
                )
            
            with col_e:
                st.metric(
                    "Acceptance Rate",
                    f"{result.acceptance_rate:.2%}",
                    delta=None
                )
            
            with col_f:
                st.metric(
                    "Execution Time",
                    f"{result.execution_time:.2f}s",
                    delta=None
                )
            
            # Show annealing progress
            st.markdown("### Annealing Progress")
            
            # Create custom figure for better control
            progress_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Energy History", "Temperature Cooling"],
                horizontal_spacing=0.1
            )
            
            # Add energy history
            progress_fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.energy_history))),
                    y=result.energy_history,
                    mode='lines',
                    name='Energy'
                ),
                row=1, col=1
            )
            
            # Add temperature history
            progress_fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.temperature_history))),
                    y=result.temperature_history,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            # Update layout
            progress_fig.update_layout(
                height=400,
                showlegend=True
            )
            
            # Update axes
            progress_fig.update_xaxes(title_text="Iteration", row=1, col=1)
            progress_fig.update_yaxes(title_text="Energy", row=1, col=1)
            progress_fig.update_xaxes(title_text="Iteration", row=1, col=2)
            progress_fig.update_yaxes(title_text="Temperature", row=1, col=2)
            
            st.plotly_chart(progress_fig, use_container_width=True)
            
            # Solution details
            st.markdown("### Solution Details")
            st.markdown(f"""
            **Initial Solution**: {array_to_str(initial_solution)}
            
            **Best Solution Found**: {array_to_str(result.best_solution)}
            
            **Global Minimum**: {array_to_str(problem["global_minimum"])}
            
            **Distance to Global Minimum**: {euclidean_distance(result.best_solution, problem["global_minimum"]):.6f}
            """)
        else:
            # Default visualization of the selected problem
            problem = problems[problem_name]
            fig = visualize_2d_function(
                problem["function"],
                problem["bounds"],
                points=[problem["global_minimum"]],
                title=f"{problem_name}"
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------
# Genetic Algorithm
# --------------------------------

# Function implements subject page
# Method creates predicate UI
# Operation displays object interface
def genetic_algorithm_page():
    """Genetic Algorithm demonstration page"""
    st.header("Genetic Algorithm Optimization")
    
    st.markdown("""
    Genetic Algorithm is an evolutionary optimization method inspired by natural selection.
    It evolves a population of candidate solutions over multiple generations, using
    mechanisms of selection, crossover, and mutation.
    
    Key characteristics:
    - Maintains a population of solutions
    - Uses selection to favor better solutions
    - Applies crossover to combine good solutions
    - Uses mutation to maintain diversity
    """)
    
    if not HAS_GENETIC_ALGORITHM:
        st.warning("Genetic Algorithm module not found. Please ensure the core.mathematics.optimization package is properly installed.")
        return
    
    # Get optimization problems
    problems = get_optimization_problems()
    
    # Create two columns for settings and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Algorithm Settings")
        
        # Problem selection
        problem_name = st.selectbox(
            "Test Problem",
            list(problems.keys()),
            index=0,
            key="ga_problem"
        )
        
        problem = problems[problem_name]
        
        with st.expander("Problem Description", expanded=False):
            st.markdown(problem["description"])
        
        # Algorithm parameters
        population_size = st.slider(
            "Population Size",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
        
        max_generations = st.slider(
            "Maximum Generations",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
        
        crossover_rate = st.slider(
            "Crossover Rate",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        mutation_rate = st.slider(
            "Mutation Rate",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01
        )
        
        selection_method = st.selectbox(
            "Selection Method",
            options=[sm.value for sm in SelectionMethod],
            index=0
        )
        
        elitism_count = st.slider(
            "Elitism Count",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )
        
        # Run optimization
        if st.button("Run Optimization", key="run_ga"):
            # Show a spinner while running
            with st.spinner("Running Genetic Algorithm..."):
                # Configure the algorithm
                config = GeneticConfig(
                    population_size=population_size,
                    max_generations=max_generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    elitism_count=elitism_count,
                    selection_method=SelectionMethod(selection_method),
                    crossover_method=CrossoverMethod.UNIFORM,
                    random_seed=42
                )
                
                # Create optimizer
                optimizer = GeneticAlgorithm(config)
                
                # Create genome operations
                class RealValuedGenomeOps:
                    def __init__(self, bounds, objective_fn, dimensions=2):
                        self.bounds = bounds
                        self.objective_fn = objective_fn
                        self.dimensions = dimensions
                    
                    def create_genome(self, random_state):
                        # Random position within bounds
                        genome = np.zeros(self.dimensions)
                        for i in range(self.dimensions):
                            genome[i] = random_state.uniform(self.bounds[i][0], self.bounds[i][1])
                        return genome
                    
                    def calculate_fitness(self, genome):
                        # Negate because we want to maximize (GA maximizes)
                        return -self.objective_fn(genome)
                    
                    def mutate(self, genome, mutation_rate, random_state):
                        # Create copy
                        mutated = genome.copy()
                        
                        # Apply mutation to each gene with probability
                        for i in range(self.dimensions):
                            if random_state.random() < mutation_rate:
                                # Gaussian mutation scaled by bounds
                                bound_range = self.bounds[i][1] - self.bounds[i][0]
                                delta = random_state.normal(0, 0.1 * bound_range)
                                mutated[i] += delta
                                mutated[i] = np.clip(mutated[i], self.bounds[i][0], self.bounds[i][1])
                        
                        return mutated
                    
                    def crossover(self, parent1, parent2, random_state):
                        # Uniform crossover
                        child1 = parent1.copy()
                        child2 = parent2.copy()
                        
                        for i in range(self.dimensions):
                            if random_state.random() < 0.5:
                                child1[i], child2[i] = child2[i], child1[i]
                        
                        return child1, child2
                    
                    def calculate_diversity(self, population):
                        # Average pairwise Euclidean distance
                        if len(population) <= 1:
                            return 0.0
                        
                        dist_sum = 0
                        count = 0
                        
                        for i in range(len(population)):
                            for j in range(i+1, len(population)):
                                dist_sum += np.sqrt(np.sum((population[i] - population[j])**2))
                                count += 1
                        
                        return dist_sum / count if count > 0 else 0.0
                
                # Create genome operations
                genome_ops = RealValuedGenomeOps(
                    bounds=problem["bounds"],
                    objective_fn=problem["function"]
                )
                
                # Run optimization
                result = optimizer.evolve(genome_ops)
                
                # Store result in session state
                st.session_state.ga_result = result
                st.session_state.ga_problem = problem
    
    with col2:
        st.subheader("Visualization")
        
        # Visualize the problem
        if "ga_result" in st.session_state:
            result = st.session_state.ga_result
            problem = st.session_state.ga_problem
            
            # Get the objective function
            objective_fn = problem["function"]
            
            # Create function visualization
            points = [
                result.best_genome,  # Best solution
                problem["global_minimum"]  # Global minimum
            ]
            
            fig = visualize_2d_function(
                objective_fn,
                problem["bounds"],
                points=points,
                title=f"{problem_name} Optimization"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show results
            st.markdown("### Optimization Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Initial Fitness",
                    f"{-result.initial_best_fitness:.6f}",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Final Fitness",
                    f"{-result.best_fitness:.6f}",
                    delta=f"{-abs(result.initial_best_fitness - result.best_fitness):.6f}",
                    delta_color="inverse"
                )
            
            with col_c:
                st.metric(
                    "Global Minimum",
                    f"{problem['global_minimum_value']:.6f}",
                    delta=f"{abs(-result.best_fitness - problem['global_minimum_value']):.6f}",
                    delta_color="inverse"
                )
            
            # Additional info
            col_d, col_e, col_f = st.columns(3)
            
            with col_d:
                st.metric(
                    "Generations",
                    f"{result.generations}",
                    delta=None
                )
            
            with col_e:
                st.metric(
                    "Population Size",
                    f"{result.population_size}",
                    delta=None
                )
            
            with col_f:
                st.metric(
                    "Execution Time",
                    f"{result.execution_time:.2f}s",
                    delta=None
                )
            
            # Show evolution progress
            st.markdown("### Evolution Progress")
            
            # Create custom figure for better control
            progress_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Fitness History", "Population Diversity"],
                horizontal_spacing=0.1
            )
            
            # Negate fitness for visualization (problem is minimization)
            fitness_history = [-f for f in result.fitness_history]
            
            # Add fitness history
            progress_fig.add_trace(
                go.Scatter(
                    x=list(range(len(fitness_history))),
                    y=fitness_history,
                    mode='lines',
                    name='Best Fitness'
                ),
                row=1, col=1
            )
            
            # Add diversity history
            progress_fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.diversity_history))),
                    y=result.diversity_history,
                    mode='lines',
                    name='Population Diversity',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Update layout
            progress_fig.update_layout(
                height=400,
                showlegend=True
            )
            
            # Update axes
            progress_fig.update_xaxes(title_text="Generation", row=1, col=1)
            progress_fig.update_yaxes(title_text="Fitness", row=1, col=1)
            progress_fig.update_xaxes(title_text="Generation", row=1, col=2)
            progress_fig.update_yaxes(title_text="Diversity", row=1, col=2)
            
            st.plotly_chart(progress_fig, use_container_width=True)
            
            # Solution details
            st.markdown("### Solution Details")
            st.markdown(f"""
            **Best Solution Found**: {array_to_str(result.best_genome)}
            
            **Global Minimum**: {array_to_str(problem["global_minimum"])}
            
            **Distance to Global Minimum**: {euclidean_distance(result.best_genome, problem["global_minimum"]):.6f}
            """)
        else:
            # Default visualization of the selected problem
            problem = problems[problem_name]
            fig = visualize_2d_function(
                problem["function"],
                problem["bounds"],
                points=[problem["global_minimum"]],
                title=f"{problem_name}"
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------
# Particle Swarm Optimization
# --------------------------------

# Function implements subject page
# Method creates predicate UI
# Operation displays object interface
def particle_swarm_page():
    """Particle Swarm Optimization demonstration page"""
    st.header("Particle Swarm Optimization")
    
    st.markdown("""
    Particle Swarm Optimization (PSO) is a population-based stochastic optimization technique
    inspired by social behavior of bird flocking or fish schooling. Each particle in the swarm
    represents a candidate solution that moves through the search space.
    
    Key characteristics:
    - Particles move based on their own best position and swarm's best position
    - Balances exploration and exploitation through inertia and acceleration coefficients
    - No crossover or mutation operators like genetic algorithms
    - Effective for continuous optimization problems
    """)
    
    if not HAS_PARTICLE_SWARM:
        st.warning("Particle Swarm Optimization module not found. Please ensure the core.mathematics.optimization package is properly installed.")
        return
    
    # Get optimization problems
    problems = get_optimization_problems()
    
    # Create two columns for settings and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Algorithm Settings")
        
        # Problem selection
        problem_name = st.selectbox(
            "Test Problem",
            list(problems.keys()),
            index=0,
            key="pso_problem"
        )
        
        problem = problems[problem_name]
        
        with st.expander("Problem Description", expanded=False):
            st.markdown(problem["description"])
        
        # Algorithm parameters
        swarm_size = st.slider(
            "Swarm Size",
            min_value=10,
            max_value=500,
            value=50,
            step=10
        )
        
        max_iterations = st.slider(
            "Maximum Iterations",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="pso_max_iter"
        )
        
        inertia_weight = st.slider(
            "Inertia Weight",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        cognitive_coefficient = st.slider(
            "Cognitive Coefficient (c1)",
            min_value=0.1,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
        
        social_coefficient = st.slider(
            "Social Coefficient (c2)",
            min_value=0.1,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
        
        topology = st.selectbox(
            "Swarm Topology",
            options=[t.value for t in TopologyType],
            index=0
        )
        
        dynamic_inertia = st.checkbox("Dynamic Inertia Weight", value=True)
        
        # Run optimization
        if st.button("Run Optimization", key="run_pso"):
            # Show a spinner while running
            with st.spinner("Running Particle Swarm Optimization..."):
                # Configure the algorithm
                config = PSOConfig(
                    swarm_size=swarm_size,
                    max_iterations=max_iterations,
                    inertia_weight=inertia_weight,
                    cognitive_coefficient=cognitive_coefficient,
                    social_coefficient=social_coefficient,
                    topology=TopologyType(topology),
                    dynamic_inertia=dynamic_inertia,
                    random_seed=42
                )
                
                # Create optimizer
                optimizer = ParticleSwarmOptimization(config)
                
                # Get the objective function (PSO maximizes by default)
                objective_fn = problem["function"]
                fitness_fn = lambda x: -objective_fn(x)  # Negate for maximization
                
                # Run optimization
                result = optimizer.optimize(
                    fitness_function=fitness_fn,
                    dimensions=2,
                    bounds=problem["bounds"],
                    maximize=True
                )
                
                # Store result in session state
                st.session_state.pso_result = result
                st.session_state.pso_problem = problem
    
    with col2:
        st.subheader("Visualization")
        
        # Visualize the problem
        if "pso_result" in st.session_state:
            result = st.session_state.pso_result
            problem = st.session_state.pso_problem
            
            # Get the objective function
            objective_fn = problem["function"]
            
            # Create function visualization
            points = [
                result.best_position,  # Best solution
                problem["global_minimum"]  # Global minimum
            ]
            
            fig = visualize_2d_function(
                objective_fn,
                problem["bounds"],
                points=points,
                title=f"{problem_name} Optimization"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show results
            st.markdown("### Optimization Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Initial Fitness",
                    f"{-result.initial_best_fitness:.6f}",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Final Fitness",
                    f"{-result.best_fitness:.6f}",
                    delta=f"{-abs(result.initial_best_fitness - result.best_fitness):.6f}",
                    delta_color="inverse"
                )
            
            with col_c:
                st.metric(
                    "Global Minimum",
                    f"{problem['global_minimum_value']:.6f}",
                    delta=f"{abs(-result.best_fitness - problem['global_minimum_value']):.6f}",
                    delta_color="inverse"
                )
            
            # Additional info
            col_d, col_e, col_f = st.columns(3)
            
            with col_d:
                st.metric(
                    "Iterations",
                    f"{result.iterations}",
                    delta=None
                )
            
            with col_e:
                st.metric(
                    "Swarm Size",
                    f"{result.swarm_size}",
                    delta=None
                )
            
            with col_f:
                st.metric(
                    "Execution Time",
                    f"{result.execution_time:.2f}s",
                    delta=None
                )
            
            # Show optimization progress
            st.markdown("### Optimization Progress")
            
            # Create custom figure for better control
            progress_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Fitness History", "Swarm Diversity"],
                horizontal_spacing=0.1
            )
            
            # Negate fitness for visualization (problem is minimization)
            fitness_history = [-f for f in result.fitness_history]
            
            # Add fitness history
            progress_fig.add_trace(
                go.Scatter(
                    x=list(range(len(fitness_history))),
                    y=fitness_history,
                    mode='lines',
                    name='Best Fitness'
                ),
                row=1, col=1
            )
            
            # Add diversity history
            progress_fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.diversity_history))),
                    y=result.diversity_history,
                    mode='lines',
                    name='Swarm Diversity',
                    line=dict(color='orange')
                ),
                row=1, col=2
            )
            
            # Update layout
            progress_fig.update_layout(
                height=400,
                showlegend=True
            )
            
            # Update axes
            progress_fig.update_xaxes(title_text="Iteration", row=1, col=1)
            progress_fig.update_yaxes(title_text="Fitness", row=1, col=1)
            progress_fig.update_xaxes(title_text="Iteration", row=1, col=2)
            progress_fig.update_yaxes(title_text="Diversity", row=1, col=2)
            
            st.plotly_chart(progress_fig, use_container_width=True)
            
            # Solution details
            st.markdown("### Solution Details")
            st.markdown(f"""
            **Best Solution Found**: {array_to_str(result.best_position)}
            
            **Global Minimum**: {array_to_str(problem["global_minimum"])}
            
            **Distance to Global Minimum**: {euclidean_distance(result.best_position, problem["global_minimum"]):.6f}
            """)
        else:
            # Default visualization of the selected problem
            problem = problems[problem_name]
            fig = visualize_2d_function(
                problem["function"],
                problem["bounds"],
                points=[problem["global_minimum"]],
                title=f"{problem_name}"
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------
# Monte Carlo Simulation
# --------------------------------

# Function implements subject page
# Method creates predicate UI
# Operation displays object interface
def monte_carlo_page():
    """Monte Carlo Simulation demonstration page"""
    st.header("Monte Carlo Simulation and Integration")
    
    st.markdown("""
    Monte Carlo methods use repeated random sampling to obtain numerical results. These
    methods are particularly useful for estimating integrals, solving differential equations,
    and uncertainty quantification in complex systems.
    
    Key applications:
    - Numerical integration in high dimensions
    - Sensitivity analysis and uncertainty propagation
    - Optimization of complex systems
    - Risk assessment and scenario analysis
    """)
    
    if not HAS_MONTE_CARLO:
        st.warning("Monte Carlo module not found. Please ensure the core.mathematics.monte_carlo package is properly installed.")
        return
    
    # Create tabs for different Monte Carlo applications
    mc_tabs = st.tabs(["Simulation", "Integration"])
    
    # Simulation tab
    with mc_tabs[0]:
        st.subheader("Monte Carlo Simulation")
        
        st.markdown("""
        This demonstration shows how Monte Carlo simulation can be used to
        estimate statistical properties of complex systems by running multiple
        randomized trials.
        """)
        
        # Create two columns for settings and visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Simulation Settings")
            
            # Simulation parameters
            iterations = st.slider(
                "Number of Iterations",
                min_value=100,
                max_value=100000,
                value=10000,
                step=100
            )
            
            simulation_type = st.selectbox(
                "Simulation Type",
                options=["Stock Price", "Project Duration", "Portfolio Value"],
                index=0
            )
            
            if simulation_type == "Stock Price":
                initial_price = st.number_input(
                    "Initial Stock Price",
                    min_value=1.0,
                    max_value=1000.0,
                    value=100.0,
                    step=10.0
                )
                
                volatility = st.slider(
                    "Annual Volatility (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0
                ) / 100.0
                
                drift = st.slider(
                    "Annual Drift (%)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=5.0,
                    step=1.0
                ) / 100.0
                
                days = st.slider(
                    "Days to Simulate",
                    min_value=1,
                    max_value=365,
                    value=30,
                    step=1
                )
                
                # Run simulation
                if st.button("Run Simulation", key="run_mc_sim"):
                    # Show a spinner while running
                    with st.spinner("Running Monte Carlo Simulation..."):
                        # Define stock price simulation function
                        def stock_price_simulation(random_state, iteration,
                                                initial_price=100.0, 
                                                volatility=0.2,
                                                drift=0.05,
                                                days=30):
                            # Daily parameters
                            daily_vol = volatility / np.sqrt(252)
                            daily_drift = drift / 252
                            
                            # Simulate price path
                            price = initial_price
                            for _ in range(days):
                                price *= np.exp((daily_drift - 0.5 * daily_vol**2) + 
                                               daily_vol * random_state.normal())
                            
                            return price
                        
                        # Create and run simulator
                        simulator = MCSimulation()
                        result = simulator.run_simulation(
                            simulation_function=stock_price_simulation,
                            iterations=iterations,
                            initial_price=initial_price,
                            volatility=volatility,
                            drift=drift,
                            days=days
                        )
                        
                        # Store result in session state
                        st.session_state.mc_sim_result = result
                        st.session_state.mc_sim_type = simulation_type
                        st.session_state.mc_sim_params = {
                            "initial_price": initial_price,
                            "volatility": volatility,
                            "drift": drift,
                            "days": days
                        }
            
            elif simulation_type == "Project Duration":
                tasks = st.slider(
                    "Number of Tasks",
                    min_value=3,
                    max_value=20,
                    value=5,
                    step=1
                )
                
                min_duration = st.slider(
                    "Minimum Task Duration (days)",
                    min_value=1,
                    max_value=30,
                    value=3,
                    step=1
                )
                
                max_duration = st.slider(
                    "Maximum Task Duration (days)",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1
                )
                
                dependency_chance = st.slider(
                    "Task Dependency Chance (%)",
                    min_value=0,
                    max_value=100,
                    value=30,
                    step=5
                ) / 100.0
                
                # Run simulation
                if st.button("Run Simulation", key="run_mc_project"):
                    # Show a spinner while running
                    with st.spinner("Running Monte Carlo Simulation..."):
                        # Define project duration simulation function
                        def project_duration_simulation(random_state, iteration,
                                                     tasks=5, 
                                                     min_duration=3,
                                                     max_duration=10,
                                                     dependency_chance=0.3):
                            # Generate task durations
                            durations = random_state.uniform(min_duration, max_duration, tasks)
                            
                            # Generate random dependency matrix (directed acyclic graph)
                            dependency_matrix = np.zeros((tasks, tasks))
                            for i in range(tasks):
                                for j in range(i+1, tasks):  # Ensure DAG
                                    if random_state.random() < dependency_chance:
                                        dependency_matrix[i, j] = 1
                            
                            # Calculate earliest completion time for each task
                            completion_times = np.zeros(tasks)
                            for i in range(tasks):
                                # Get all dependencies
                                deps = np.where(dependency_matrix[:, i] == 1)[0]
                                if len(deps) > 0:
                                    # Task starts after all dependencies are complete
                                    completion_times[i] = max(completion_times[deps]) + durations[i]
                                else:
                                    # Task has no dependencies
                                    completion_times[i] = durations[i]
                            
                            # Project completion is the maximum completion time
                            return max(completion_times)
                        
                        # Create and run simulator
                        simulator = MCSimulation()
                        result = simulator.run_simulation(
                            simulation_function=project_duration_simulation,
                            iterations=iterations,
                            tasks=tasks,
                            min_duration=min_duration,
                            max_duration=max_duration,
                            dependency_chance=dependency_chance
                        )
                        
                        # Store result in session state
                        st.session_state.mc_sim_result = result
                        st.session_state.mc_sim_type = simulation_type
                        st.session_state.mc_sim_params = {
                            "tasks": tasks,
                            "min_duration": min_duration,
                            "max_duration": max_duration,
                            "dependency_chance": dependency_chance
                        }
            
            elif simulation_type == "Portfolio Value":
                assets = st.slider(
                    "Number of Assets",
                    min_value=2,
                    max_value=10,
                    value=4,
                    step=1
                )
                
                initial_investment = st.number_input(
                    "Initial Investment",
                    min_value=1000.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=1000.0
                )
                
                years = st.slider(
                    "Simulation Years",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1
                )
                
                avg_return = st.slider(
                    "Average Annual Return (%)",
                    min_value=0.0,
                    max_value=30.0,
                    value=8.0,
                    step=0.5
                ) / 100.0
                
                avg_volatility = st.slider(
                    "Average Volatility (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=15.0,
                    step=1.0
                ) / 100.0
                
                # Run simulation
                if st.button("Run Simulation", key="run_mc_portfolio"):
                    # Show a spinner while running
                    with st.spinner("Running Monte Carlo Simulation..."):
                        # Define portfolio simulation function
                        def portfolio_simulation(random_state, iteration,
                                              assets=4, 
                                              initial_investment=10000.0,
                                              years=5,
                                              avg_return=0.08,
                                              avg_volatility=0.15):
                            # Generate random asset returns and volatilities
                            returns = avg_return + random_state.normal(0, 0.05, assets)
                            volatilities = avg_volatility + random_state.normal(0, 0.03, assets)
                            volatilities = np.abs(volatilities)  # Ensure positive volatilities
                            
                            # Generate random correlation matrix
                            correlations = random_state.uniform(0.1, 0.7, (assets, assets))
                            corr_matrix = np.eye(assets)
                            for i in range(assets):
                                for j in range(i+1, assets):
                                    corr_matrix[i, j] = correlations[i, j]
                                    corr_matrix[j, i] = correlations[i, j]
                            
                            # Generate random weights (summing to 1)
                            weights = random_state.random(assets)
                            weights = weights / np.sum(weights)
                            
                            # Calculate portfolio parameters
                            portfolio_return = np.sum(returns * weights)
                            
                            # Create covariance matrix
                            cov_matrix = np.zeros((assets, assets))
                            for i in range(assets):
                                for j in range(assets):
                                    cov_matrix[i, j] = volatilities[i] * volatilities[j] * corr_matrix[i, j]
                            
                            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                            portfolio_volatility = np.sqrt(portfolio_variance)
                            
                            # Simulate portfolio growth
                            value = initial_investment
                            for _ in range(years):
                                annual_return = random_state.normal(portfolio_return, portfolio_volatility)
                                value *= (1 + annual_return)
                            
                            return value
                        
                        # Create and run simulator
                        simulator = MCSimulation()
                        result = simulator.run_simulation(
                            simulation_function=portfolio_simulation,
                            iterations=iterations,
                            assets=assets,
                            initial_investment=initial_investment,
                            years=years,
                            avg_return=avg_return,
                            avg_volatility=avg_volatility
                        )
                        
                        # Store result in session state
                        st.session_state.mc_sim_result = result
                        st.session_state.mc_sim_type = simulation_type
                        st.session_state.mc_sim_params = {
                            "assets": assets,
                            "initial_investment": initial_investment,
                            "years": years,
                            "avg_return": avg_return,
                            "avg_volatility": avg_volatility
                        }
        
        with col2:
            st.subheader("Simulation Results")
            
            # Visualize simulation results
            if "mc_sim_result" in st.session_state:
                result = st.session_state.mc_sim_result
                sim_type = st.session_state.mc_sim_type
                params = st.session_state.mc_sim_params
                
                # Display distribution of simulation results
                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=["Histogram of Results", "Percentile Analysis"],
                    vertical_spacing=0.1
                )
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=result.raw_data,
                        nbinsx=50,
                        name="Frequency",
                        marker_color="steelblue",
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Add vertical lines for key statistics
                fig.add_vline(
                    x=result.mean, 
                    line_width=2, 
                    line_dash="dash", 
                    line_color="red",
                    row=1, col=1
                )
                
                fig.add_vline(
                    x=result.median, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color="green",
                    row=1, col=1
                )
                
                # Get percentiles
                percentiles = result.percentiles
                percentile_vals = [percentiles[5], percentiles[25], percentiles[50], 
                                  percentiles[75], percentiles[95]]
                
                # Add percentile analysis
                fig.add_trace(
                    go.Scatter(
                        x=percentile_vals,
                        y=[1] * len(percentile_vals),
                        mode="markers+text",
                        marker=dict(
                            color="purple",
                            size=15,
                            symbol="diamond"
                        ),
                        text=["5%", "25%", "50%", "75%", "95%"],
                        textposition="top center",
                        name="Percentiles"
                    ),
                    row=2, col=1
                )
                
                # Add range lines
                fig.add_shape(
                    type="line",
                    x0=percentiles[5],
                    x1=percentiles[95],
                    y0=1,
                    y1=1,
                    line=dict(
                        color="purple",
                        width=3
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=600,
                    showlegend=True
                )
                
                # Update axes
                if sim_type == "Stock Price":
                    x_title = "Final Stock Price"
                elif sim_type == "Project Duration":
                    x_title = "Project Duration (days)"
                else:  # Portfolio Value
                    x_title = "Final Portfolio Value"
                
                fig.update_xaxes(title_text=x_title, row=1, col=1)
                fig.update_yaxes(title_text="Frequency", row=1, col=1)
                fig.update_xaxes(title_text=x_title, row=2, col=1)
                fig.update_yaxes(showticklabels=False, row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics
                st.markdown("### Summary Statistics")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Mean",
                        f"{result.mean:.2f}",
                        delta=None
                    )
                
                with col_b:
                    st.metric(
                        "Median",
                        f"{result.median:.2f}",
                        delta=None
                    )
                
                with col_c:
                    st.metric(
                        "Standard Deviation",
                        f"{result.std_dev:.2f}",
                        delta=None
                    )
                
                # Additional info
                st.markdown("### Percentile Analysis")
                
                col_d, col_e, col_f, col_g, col_h = st.columns(5)
                
                with col_d:
                    st.metric(
                        "5th Percentile",
                        f"{percentiles[5]:.2f}",
                        delta=None
                    )
                
                with col_e:
                    st.metric(
                        "25th Percentile",
                        f"{percentiles[25]:.2f}",
                        delta=None
                    )
                
                with col_f:
                    st.metric(
                        "50th Percentile",
                        f"{percentiles[50]:.2f}",
                        delta=None
                    )
                
                with col_g:
                    st.metric(
                        "75th Percentile",
                        f"{percentiles[75]:.2f}",
                        delta=None
                    )
                
                with col_h:
                    st.metric(
                        "95th Percentile",
                        f"{percentiles[95]:.2f}",
                        delta=None
                    )
                
                # Display simulation parameters
                st.markdown("### Simulation Parameters")
                param_text = ""
                
                if sim_type == "Stock Price":
                    param_text = f"""
                    - Initial Price: ${params['initial_price']:.2f}
                    - Annual Volatility: {params['volatility']*100:.1f}%
                    - Annual Drift (Return): {params['drift']*100:.1f}%
                    - Days Simulated: {params['days']}
                    """
                
                elif sim_type == "Project Duration":
                    param_text = f"""
                    - Number of Tasks: {params['tasks']}
                    - Task Duration Range: {params['min_duration']} to {params['max_duration']} days
                    - Task Dependency Probability: {params['dependency_chance']*100:.1f}%
                    """
                
                elif sim_type == "Portfolio Value":
                    param_text = f"""
                    - Number of Assets: {params['assets']}
                    - Initial Investment: ${params['initial_investment']:.2f}
                    - Simulation Years: {params['years']}
                    - Average Annual Return: {params['avg_return']*100:.1f}%
                    - Average Volatility: {params['avg_volatility']*100:.1f}%
                    """
                
                st.markdown(param_text)
                
                # Display iterations and execution time
                st.markdown(f"""
                **Number of Iterations**: {result.iterations:,}
                
                **Execution Time**: {result.metadata['execution_time']:.3f} seconds
                """)
            else:
                st.info("Run a simulation to view results")
    
    # Integration tab
    with mc_tabs[1]:
        st.subheader("Monte Carlo Integration")
        
        st.markdown("""
        Monte Carlo integration uses random sampling to estimate the value of integrals,
        particularly useful for high-dimensional problems where traditional numerical 
        integration methods become inefficient.
        """)
        
        # Create two columns for settings and visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Integration Settings")
            
            function_name = st.selectbox(
                "Function to Integrate",
                options=["Circle Area (π estimation)", "Gaussian Function", "Sine + Exponential"],
                index=0
            )
            
            samples = st.slider(
                "Number of Samples",
                min_value=100,
                max_value=1000000,
                value=10000,
                step=100
            )
            
            method = st.selectbox(
                "Integration Method",
                options=["Uniform Sampling", "Stratified Sampling"],
                index=0
            )
            
            # Run integration
            if st.button("Run Integration", key="run_mc_int"):
                # Show a spinner while running
                with st.spinner("Running Monte Carlo Integration..."):
                    # Configure and run the integration
                    integrator = MCIntegration()
                    
                    if function_name == "Circle Area (π estimation)":
                        # Function f(x,y) = 1 if x²+y² <= 1, else 0
                        # Integral over [0,1] × [0,1] is π/4
                        
                        def circle_indicator(x, y):
                            return 1.0 if x**2 + y**2 <= 1.0 else 0.0
                        
                        domain = [(0, 1), (0, 1)]
                        true_value = np.pi / 4
                        
                        if method == "Uniform Sampling":
                            result = integrator.integrate_uniform(
                                func=circle_indicator,
                                domain=domain,
                                iterations=samples
                            )
                        else:  # Stratified sampling
                            strata_per_dim = min(100, int(np.sqrt(samples / 5)))
                            result = integrator.stratified_sampling(
                                func=circle_indicator,
                                domain=domain,
                                strata_per_dim=strata_per_dim,
                                samples_per_stratum=max(1, samples // (strata_per_dim**2))
                            )
                    
                    elif function_name == "Gaussian Function":
                        # Function f(x,y) = exp(-(x²+y²)/2)/(2π)
                        # Integral over [-5,5] × [-5,5] is close to 1
                        
                        def gaussian(x, y):
                            return np.exp(-(x**2 + y**2) / 2) / (2 * np.pi)
                        
                        domain = [(-5, 5), (-5, 5)]
                        true_value = 1.0  # Approximately
                        
                        if method == "Uniform Sampling":
                            result = integrator.integrate_uniform(
                                func=gaussian,
                                domain=domain,
                                iterations=samples
                            )
                        else:  # Stratified sampling
                            strata_per_dim = min(100, int(np.sqrt(samples / 5)))
                            result = integrator.stratified_sampling(
                                func=gaussian,
                                domain=domain,
                                strata_per_dim=strata_per_dim,
                                samples_per_stratum=max(1, samples // (strata_per_dim**2))
                            )
                    
                    else:  # Sine + Exponential
                        # Function f(x,y) = sin(x) * exp(-y²)
                        # No simple analytical solution
                        
                        def sine_exp(x, y):
                            return np.sin(x) * np.exp(-y**2)
                        
                        domain = [(0, np.pi), (-2, 2)]
                        true_value = 2.0  # Approximation for this example
                        
                        if method == "Uniform Sampling":
                            result = integrator.integrate_uniform(
                                func=sine_exp,
                                domain=domain,
                                iterations=samples
                            )
                        else:  # Stratified sampling
                            strata_per_dim = min(100, int(np.sqrt(samples / 5)))
                            result = integrator.stratified_sampling(
                                func=sine_exp,
                                domain=domain,
                                strata_per_dim=strata_per_dim,
                                samples_per_stratum=max(1, samples // (strata_per_dim**2))
                            )
                    
                    # Store result in session state
                    st.session_state.mc_int_result = result
                    st.session_state.mc_int_function = function_name
                    st.session_state.mc_int_method = method
                    st.session_state.mc_int_true_value = true_value
        
        with col2:
            st.subheader("Integration Results")
            
            # Visualize integration results
            if "mc_int_result" in st.session_state:
                result = st.session_state.mc_int_result
                function = st.session_state.mc_int_function
                method = st.session_state.mc_int_method
                true_value = st.session_state.mc_int_true_value
                
                # Show results
                st.markdown("### Integration Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Estimated Value",
                        f"{result.integral_value:.6f}",
                        delta=None
                    )
                
                with col_b:
                    st.metric(
                        "Error Estimate",
                        f"±{result.error_estimate:.6f}",
                        delta=None
                    )
                
                with col_c:
                    st.metric(
                        "True Value",
                        f"{true_value:.6f}",
                        delta=f"{abs(result.integral_value - true_value):.6f}",
                        delta_color="inverse"
                    )
                
                # Visualization based on integration function
                if function == "Circle Area (π estimation)":
                    # Create visualization of Monte Carlo for circle area
                    # Generate random points
                    rng = np.random.RandomState(42)
                    num_points = min(10000, result.iterations)
                    points_x = rng.uniform(0, 1, num_points)
                    points_y = rng.uniform(0, 1, num_points)
                    
                    # Determine which points are inside circle
                    distances = np.sqrt(points_x**2 + points_y**2)
                    inside = distances <= 1.0
                    
                    # Create scatter plot
                    fig = go.Figure()
                    
                    # Add circle outline
                    theta = np.linspace(0, np.pi/2, 100)
                    x_circle = np.cos(theta)
                    y_circle = np.sin(theta)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_circle,
                            y=y_circle,
                            mode="lines",
                            line=dict(color="black", width=2),
                            name="Unit Circle"
                        )
                    )
                    
                    # Add points inside circle
                    fig.add_trace(
                        go.Scatter(
                            x=points_x[inside],
                            y=points_y[inside],
                            mode="markers",
                            marker=dict(
                                color="green",
                                size=3,
                                opacity=0.5
                            ),
                            name="Inside Circle"
                        )
                    )
                    
                    # Add points outside circle
                    fig.add_trace(
                        go.Scatter(
                            x=points_x[~inside],
                            y=points_y[~inside],
                            mode="markers",
                            marker=dict(
                                color="red",
                                size=3,
                                opacity=0.5
                            ),
                            name="Outside Circle"
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Monte Carlo Estimation of π (showing {num_points:,} of {result.iterations:,} points)",
                        xaxis_title="x",
                        yaxis_title="y",
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        height=500
                    )
                    
                    # Add annotations
                    inside_count = np.sum(inside)
                    fig.add_annotation(
                        x=0.5,
                        y=0.1,
                        text=f"π estimate: 4 × ({inside_count}/{num_points}) = {4 * inside_count / num_points:.6f}",
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Generic visualization of convergence
                    # Create a plot showing convergence over iterations
                    samples_sequence = np.logspace(2, np.log10(result.iterations), 20).astype(int)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add true value line
                    fig.add_hline(
                        y=true_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="True Value",
                        annotation_position="bottom right"
                    )
                    
                    # Add confidence interval
                    fig.add_hrect(
                        y0=result.integral_value - result.error_estimate,
                        y1=result.integral_value + result.error_estimate,
                        line_width=0,
                        fillcolor="gray",
                        opacity=0.2,
                        annotation_text="Error Estimate",
                        annotation_position="bottom left"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Monte Carlo Integration Convergence ({function})",
                        xaxis_title="Number of Samples (log scale)",
                        yaxis_title="Integral Estimate",
                        xaxis=dict(type="log"),
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display performance information
                st.markdown("### Performance Information")
                
                col_d, col_e, col_f = st.columns(3)
                
                with col_d:
                    st.metric(
                        "Number of Samples",
                        f"{result.iterations:,}",
                        delta=None
                    )
                
                with col_e:
                    st.metric(
                        "Execution Time",
                        f"{result.execution_time:.3f}s",
                        delta=None
                    )
                
                with col_f:
                    st.metric(
                        "Dimensions",
                        f"{result.dimension}",
                        delta=None
                    )
                
                # Display method details
                st.markdown(f"""
                ### Method: {method}
                
                **Relative Error**: {abs(result.integral_value - true_value) / abs(true_value) * 100:.4f}%
                
                **Standard Error**: {result.error_estimate:.6f}
                
                **Confidence Interval (95%)**: [{result.integral_value - 1.96 * result.error_estimate:.6f}, 
                {result.integral_value + 1.96 * result.error_estimate:.6f}]
                """)
            else:
                st.info("Run an integration to view results")


# --------------------------------
# Las Vegas Algorithm
# --------------------------------

# Function implements subject page
# Method creates predicate UI
# Operation displays object interface
def las_vegas_page():
    """Las Vegas Algorithm demonstration page"""
    st.header("Las Vegas Randomized Algorithm")
    
    st.markdown("""
    Las Vegas algorithms are randomized algorithms that always produce the correct 
    result but have non-deterministic running time. Unlike Monte Carlo algorithms, 
    which may produce incorrect results, Las Vegas algorithms guarantee correctness 
    but may take varying amounts of time to complete.
    
    Key characteristics:
    - Always produces correct results (when it terminates)
    - Running time varies based on random choices
    - May be restarted if taking too long
    - Used for problems where verification is easier than finding a solution
    """)
    
    if not HAS_LAS_VEGAS:
        st.warning("Las Vegas Algorithm module not found. Please ensure the core.mathematics.las_vegas package is properly installed.")
        return
    
    # Create two columns for settings and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Algorithm Settings")
        
        # Problem selection
        problem_type = st.selectbox(
            "Problem Type",
            options=["Find Password Hash", "Graph Coloring", "Constraint Satisfaction"],
            index=0
        )
        
        if problem_type == "Find Password Hash":
            difficulty = st.slider(
                "Difficulty Level",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                help="Higher values create more complex passwords to find"
            )
            
            max_iterations = st.slider(
                "Maximum Iterations",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            parallel = st.checkbox("Use Parallel Processing", value=True)
            
            # Run algorithm
            if st.button("Run Algorithm", key="run_lv"):
                # Show a spinner while running
                with st.spinner("Running Las Vegas Algorithm..."):
                    # Configure the algorithm
                    config = LasVegasConfig(
                        max_iterations=max_iterations,
                        parallel=parallel,
                        random_seed=42
                    )
                    
                    # Create algorithm instance
                    lv_algorithm = LasVegasAlgorithm(config)
                    
                    # Define search function to find password hash match
                    # This is a simplified hash finding problem where
                    # we're looking for a string that hashes to a value
                    # with a specific pattern (leading zeros)
                    
                    import hashlib
                    
                    # Create target pattern (e.g., hash with n leading zeros)
                    leading_zeros = difficulty
                    target_pattern = '0' * leading_zeros
                    
                    def find_hash_match(random_state, iteration, charset="abcdefghijklmnopqrstuvwxyz0123456789"):
                        # Generate random string (6-10 characters)
                        length = random_state.randint(6, 10)
                        password = ''.join(random_state.choice(list(charset)) for _ in range(length))
                        
                        # Calculate hash
                        hash_value = hashlib.md5(password.encode()).hexdigest()
                        
                        # Check if it starts with the target pattern
                        if hash_value.startswith(target_pattern):
                            return {
                                "password": password,
                                "hash": hash_value,
                                "iteration": iteration
                            }
                        
                        # No match found
                        return None
                    
                    # Run Las Vegas algorithm
                    start_time = time.time()
                    result = lv_algorithm.run(find_hash_match)
                    execution_time = time.time() - start_time
                    
                    # Store result and problem details
                    st.session_state.lv_result = result
                    st.session_state.lv_problem = {
                        "type": problem_type,
                        "difficulty": difficulty,
                        "leading_zeros": leading_zeros,
                        "target_pattern": target_pattern,
                        "max_iterations": max_iterations,
                        "parallel": parallel,
                        "execution_time": execution_time
                    }
        
        elif problem_type == "Graph Coloring":
            nodes = st.slider(
                "Number of Nodes",
                min_value=5,
                max_value=30,
                value=10,
                step=1
            )
            
            edge_probability = st.slider(
                "Edge Probability",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1
            )
            
            colors = st.slider(
                "Number of Colors",
                min_value=2,
                max_value=8,
                value=3,
                step=1
            )
            
            max_iterations = st.slider(
                "Maximum Iterations",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                key="gc_max_iter"
            )
            
            adaptive_restart = st.checkbox("Use Adaptive Restart", value=True)
            
            # Run algorithm
            if st.button("Run Algorithm", key="run_lv_gc"):
                # Show a spinner while running
                with st.spinner("Running Las Vegas Algorithm..."):
                    # Configure the algorithm
                    config = LasVegasConfig(
                        max_iterations=max_iterations,
                        random_seed=42
                    )
                    
                    # Create algorithm instance
                    lv_algorithm = LasVegasAlgorithm(config)
                    
                    # Create random graph
                    import networkx as nx
                    
                    G = nx.erdos_renyi_graph(nodes, edge_probability, seed=42)
                    
                    def graph_coloring_search(random_state, iteration):
                        # Assign random colors to nodes
                        node_colors = {}
                        for node in G.nodes():
                            node_colors[node] = random_state.randint(0, colors - 1)
                        
                        # Check if coloring is valid
                        valid = True
                        for u, v in G.edges():
                            if node_colors[u] == node_colors[v]:
                                valid = False
                                break
                        
                        if valid:
                            return {
                                "coloring": node_colors,
                                "graph": G,
                                "iteration": iteration
                            }
                        
                        # No valid coloring found
                        return None
                    
                    # Run Las Vegas algorithm
                    start_time = time.time()
                    
                    if adaptive_restart:
                        result = lv_algorithm.adaptive_restart(
                            graph_coloring_search,
                            initial_iterations=1000,
                            restart_factor=2.0,
                            max_restarts=5
                        )
                    else:
                        result = lv_algorithm.run(graph_coloring_search)
                    
                    execution_time = time.time() - start_time
                    
                    # Store result and problem details
                    st.session_state.lv_result = result
                    st.session_state.lv_problem = {
                        "type": problem_type,
                        "nodes": nodes,
                        "edge_probability": edge_probability,
                        "colors": colors,
                        "graph": G,
                        "max_iterations": max_iterations,
                        "adaptive_restart": adaptive_restart,
                        "execution_time": execution_time
                    }
    
    with col2:
        st.subheader("Algorithm Results")
        
        # Visualize algorithm results
        if "lv_result" in st.session_state:
            result = st.session_state.lv_result
            problem = st.session_state.lv_problem
            
            if result.success:
                st.success("Solution found!")
                
                if problem["type"] == "Find Password Hash":
                    st.markdown("### Password Hash Match Found")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(
                            "Iterations Required",
                            f"{result.iterations:,}",
                            delta=None
                        )
                    
                    with col_b:
                        st.metric(
                            "Execution Time",
                            f"{problem['execution_time']:.3f}s",
                            delta=None
                        )
                    
                    # Display the found password and hash
                    solution = result.solution
                    
                    st.markdown(f"""
                    **Found Password**: `{solution['password']}`
                    
                    **MD5 Hash**: `{solution['hash']}`
                    
                    **Target Pattern**: `{problem['target_pattern']}...`
                    
                    **Difficulty**: {problem['difficulty']} leading zeros
                    """)
                    
                    # Calculate and display statistics
                    avg_time = result.average_attempt_time * 1000  # Convert to ms
                    success_rate = result.success_rate * 100  # Convert to percentage
                    
                    st.markdown(f"""
                    ### Performance Statistics
                    
                    **Average Attempt Time**: {avg_time:.2f} ms
                    
                    **Success Rate**: {success_rate:.8f}%
                    
                    **Expected Iterations**: {1.0 / success_rate * 100:.0f} (1/probability)
                    """)
                    
                    # Create visualization of the search space
                    # This is a simplified visualization showing the distribution
                    # of attempt times and success probability
                    
                    # Plot attempt times
                    attempt_times = [t * 1000 for success, t in result.attempts if not success]  # Convert to ms
                    
                    if len(attempt_times) > 0:
                        fig = make_subplots(
                            rows=1, cols=1,
                            subplot_titles=["Distribution of Attempt Times"],
                        )
                        
                        fig.add_trace(
                            go.Histogram(
                                x=attempt_times,
                                nbinsx=30,
                                marker_color="steelblue",
                                opacity=0.7,
                                name="Failed Attempts"
                            )
                        )
                        
                        # Add vertical line for average time
                        fig.add_vline(
                            x=avg_time,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Average Time",
                            annotation_position="top right"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            xaxis_title="Attempt Time (ms)",
                            yaxis_title="Frequency"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif problem["type"] == "Graph Coloring":
                    st.markdown("### Valid Graph Coloring Found")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(
                            "Iterations Required",
                            f"{result.iterations:,}",
                            delta=None
                        )
                    
                    with col_b:
                        st.metric(
                            "Execution Time",
                            f"{problem['execution_time']:.3f}s",
                            delta=None
                        )
                    
                    # Display graph coloring
                    solution = result.solution
                    G = solution["graph"]
                    coloring = solution["coloring"]
                    
                    # Create node color map
                    color_map = {
                        0: "red", 1: "blue", 2: "green", 3: "yellow",
                        4: "purple", 5: "cyan", 6: "magenta", 7: "orange"
                    }
                    
                    node_colors = [color_map[coloring[n]] for n in G.nodes()]
                    
                    # Plot graph with coloring
                    import networkx as nx
                    
                    plt.figure(figsize=(8, 6))
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw(
                        G, pos, 
                        node_color=node_colors,
                        with_labels=True,
                        node_size=500,
                        font_color="white",
                        font_weight="bold",
                        edge_color="gray"
                    )
                    
                    # Save figure to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    plt.close()
                    buf.seek(0)
                    
                    # Display image
                    st.image(buf, caption="Graph Coloring Solution")
                    
                    # Display additional information
                    st.markdown(f"""
                    ### Problem Information
                    
                    **Number of Nodes**: {problem['nodes']}
                    
                    **Number of Edges**: {G.number_of_edges()}
                    
                    **Edge Density**: {G.number_of_edges() / (problem['nodes'] * (problem['nodes'] - 1) / 2):.2f}
                    
                    **Colors Used**: {problem['colors']}
                    """)
                    
                    # Calculate and display statistics
                    avg_time = result.average_attempt_time * 1000  # Convert to ms
                    success_rate = result.success_rate * 100  # Convert to percentage
                    
                    st.markdown(f"""
                    ### Performance Statistics
                    
                    **Average Attempt Time**: {avg_time:.2f} ms
                    
                    **Success Rate**: {success_rate:.8f}%
                    
                    **Expected Iterations**: {1.0 / success_rate * 100:.0f} (1/probability)
                    """)
            else:
                st.error("No solution found within the maximum iterations.")
                
                st.markdown(f"""
                ### Algorithm Statistics
                
                **Iterations Performed**: {result.iterations:,}
                
                **Execution Time**: {problem['execution_time']:.3f} seconds
                
                **Maximum Iterations**: {problem['max_iterations']:,}
                """)
        else:
            st.info("Run an algorithm to view results")


# --------------------------------
# Utility Functions
# --------------------------------

def array_to_str(arr: np.ndarray) -> str:
    """Convert numpy array to formatted string"""
    return f"[{', '.join([f'{x:.4f}' for x in arr])}]"


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((a - b) ** 2))


# --------------------------------
# Main Function
# --------------------------------

# Function runs subject application
# Method initializes predicate state
# Operation creates object interface
def main():
    """Main function for the dashboard"""
    # Configure the page
    page_config()
    
    # Initialize the dashboard
    initialize_page()
    
    # Create sidebar for algorithm selection
    st.sidebar.title("Algorithm Selection")
    
    # Create algorithm options
    algorithm_options = {
        "Simulated Annealing": simulated_annealing_page,
        "Genetic Algorithm": genetic_algorithm_page,
        "Particle Swarm Optimization": particle_swarm_page,
        "Monte Carlo Methods": monte_carlo_page,
        "Las Vegas Algorithm": las_vegas_page
    }
    
    # Add algorithm selector to sidebar
    selected_algorithm = st.sidebar.radio(
        "Select Algorithm",
        list(algorithm_options.keys())
    )
    
    # Add sidebar info
    with st.sidebar.expander("About CTAS Optimization"):
        st.markdown("""
        The CTAS (Convergent Threat Analysis System) optimization framework 
        provides advanced mathematical tools for solving complex problems in 
        intelligence analysis and decision support.
        
        These algorithms form the computational backbone of NyxTrace's 
        analytical capabilities, enabling sophisticated data processing,
        pattern recognition, and predictive modeling.
        """)
    
    # Display selected algorithm page
    algorithm_options[selected_algorithm]()
    
    # Add footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center;'>"
        f"CTAS Optimization Framework | {datetime.now().strftime('%Y-%m-%d')}"
        f"</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()