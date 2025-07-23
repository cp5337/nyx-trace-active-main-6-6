"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MONTE-CARLO-PACKAGE-0001            │
// │ 📁 domain       : Mathematics, Monte Carlo                  │
// │ 🧠 description  : Monte Carlo simulation methods package    │
// │                  for stochastic approximations              │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_MATHEMATICS                         │
// │ 🧩 dependencies : numpy, scipy                             │
// │ 🔧 tool_usage   : Simulation, Approximation                │
// │ 📡 input_type   : Mathematical functions, probability dist. │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : approximation, prediction                 │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Monte Carlo Methods Package
------------------------
This package provides implementations of Monte Carlo methods
for stochastic simulation, numerical integration, and uncertainty
quantification in intelligence analysis.
"""

from core.mathematics.monte_carlo.simulation import MCSimulation
from core.mathematics.monte_carlo.integration import MCIntegration
from core.mathematics.monte_carlo.markov_chain import MarkovChainMC
from core.mathematics.monte_carlo.uncertainty import UncertaintyPropagation

__all__ = [
    "MCSimulation",
    "MCIntegration",
    "MarkovChainMC",
    "UncertaintyPropagation",
]
