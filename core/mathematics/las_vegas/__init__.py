"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-LAS-VEGAS-PACKAGE-0001              │
// │ 📁 domain       : Mathematics, Randomized Algorithms        │
// │ 🧠 description  : Las Vegas algorithm implementation for    │
// │                  randomized search with guaranteed results  │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_MATHEMATICS                         │
// │ 🧩 dependencies : numpy, time, logging                     │
// │ 🔧 tool_usage   : Optimization, Search                     │
// │ 📡 input_type   : Problem specifications                   │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : optimization, problem-solving             │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Las Vegas Algorithm Package
------------------------
Implementation of Las Vegas randomized algorithms that always
give the correct result but have probabilistic running time.
These algorithms are useful for solving complex optimization
problems in intelligence analysis.
"""

from core.mathematics.las_vegas.algorithm import LasVegasAlgorithm
from core.mathematics.las_vegas.framework import (
    LasVegasSolution,
    LasVegasConfig,
)

__all__ = ["LasVegasAlgorithm", "LasVegasSolution", "LasVegasConfig"]
