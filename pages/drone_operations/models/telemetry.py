"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-DRONE-MODELS-TELEMETRY-0001         │
// │ 📁 domain       : Drone, Models, Telemetry                  │
// │ 🧠 description  : Data models for drone telemetry           │
// │                  with strong typing and immutability        │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DRONE_OPERATIONS                     │
// │ 🧩 dependencies : dataclasses, typing                       │
// │ 🔧 tool_usage   : Data Modeling                            │
// │ 📡 input_type   : Telemetry data                           │
// │ 🧪 test_status  : stable                                    │
// │ 🧠 cognitive_fn : data modeling, type safety                │
// │ ⌛ TTL Policy   : 6.5 Persistent                            │
// └─────────────────────────────────────────────────────────────┘

Drone Telemetry Models
---------------------
This module contains data models for drone telemetry, defining the structure 
and types for location, sensor readings, and resource status. Models follow
Rust-like principles with immutability and strong typing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

@dataclass(frozen=True)
class DroneLocation:
    """
    Immutable model for drone location data
    
    # Class defines subject location
    # Model stores predicate coordinates
    # Structure contains object position
    """
    latitude: float
    longitude: float
    altitude: float
    heading: float
    speed: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class ResourceStatus:
    """
    Immutable model for drone resource status
    
    # Class defines subject resources
    # Model stores predicate levels
    # Structure contains object status
    """
    battery_percent: float
    fuel_percent: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    storage_free_mb: float = 1000.0
    temperature_c: float = 25.0
    
    def is_critical(self) -> bool:
        """
        Check if any resource is at critical level
        
        # Function checks subject status
        # Method evaluates predicate criticality
        # Operation detects object threshold
        
        Returns:
            True if any resource is critical
        """
        return (self.battery_percent < 15.0 or 
                (self.fuel_percent is not None and self.fuel_percent < 10.0) or
                self.temperature_c > 80.0)

@dataclass(frozen=True)
class TelemetryReading:
    """
    Immutable model for complete telemetry reading
    
    # Class defines subject telemetry
    # Model stores predicate readings
    # Structure contains object measurements
    """
    drone_id: str
    location: DroneLocation
    resources: ResourceStatus
    status: str
    mission_id: Optional[str] = None
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

# CTAS Module Line Count:
# USIM Header: 17 lines
# Comments: 23 lines
# Code: 57 lines
# Total: 97 lines