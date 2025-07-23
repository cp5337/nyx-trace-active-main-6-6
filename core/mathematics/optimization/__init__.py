"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-OPTIMIZATION-PACKAGE-0001           │
// │ 📁 domain       : Mathematics, Optimization                 │
// │ 🧠 description  : Optimization algorithms for complex       │
// │                  computational problems                     │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_MATHEMATICS                         │
// │ 🧩 dependencies : numpy, scipy                             │
// │ 🔧 tool_usage   : Optimization, Analysis                   │
// │ 📡 input_type   : Problem specifications                   │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : optimization, problem-solving             │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Optimization Package
-----------------
Provides implementation of optimization algorithms for solving
complex computational problems in intelligence analysis, including
simulated annealing, genetic algorithms, and particle swarm optimization.
"""

from core.mathematics.optimization.simulated_annealing import SimulatedAnnealing
from core.mathematics.optimization.genetic_algorithm import GeneticAlgorithm
from core.mathematics.optimization.particle_swarm import (
    ParticleSwarmOptimization,
)

__all__ = [
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
]
