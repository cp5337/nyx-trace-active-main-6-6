"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-TASK-LOADER-0001                    │
// │ 📁 domain       : Data Loading, Task Management            │
// │ 🧠 description  : Loader for adversary task data            │
// │                  from JSON files                           │
// │ 🕸️ hash_type    : UUID → CUID-linked module                 │
// │ 🔄 parent_node  : NODE_DATALOADING                        │
// │ 🧩 dependencies : json, os, uuid                          │
// │ 🔧 tool_usage   : Data Loading, Initialization             │
// │ 📡 input_type   : JSON files                               │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : data loading, conversion                  │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Task Loader
----------
This module provides functionality for loading adversary task data
from JSON files into AdversaryTask objects for use in the CTAS
Periodic Table of Nodes.
"""

import json
import os
import uuid
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .adversary_task import AdversaryTask

# Configure logger
logger = logging.getLogger("periodic_table.task_loader")
logger.setLevel(logging.INFO)

# Task category colors
TASK_COLORS = {
    "Planning": "#4285F4",  # Blue
    "Reconnaissance": "#34A853",  # Green
    "Security": "#FBBC05",  # Yellow
    "Resources": "#EA4335",  # Red
    "Access": "#9C27B0",  # Purple
    "Network": "#00ACC1",  # Cyan
    "Cyber": "#FF9800",  # Orange
    "Physical": "#607D8B",  # Blue Gray
    "Execution": "#3F51B5",  # Indigo
    "Evasion": "#8BC34A",  # Light Green
    "Transnational": "#E91E63",  # Pink
}

# Default atomic properties for tasks
DEFAULT_ATOMIC_PROPERTIES = {
    "Planning": {
        "atomic_mass": 3.0,
        "valence": 3,
        "electronegativity": 2.5,
        "preferred_state": "Liquid",
        "energy_level": "ground",
    },
    "Reconnaissance": {
        "atomic_mass": 2.5,
        "valence": 5,
        "electronegativity": 3.0,
        "preferred_state": "Gas",
        "energy_level": "excited",
    },
    "Security": {
        "atomic_mass": 4.0,
        "valence": 4,
        "electronegativity": 3.2,
        "preferred_state": "Solid",
        "energy_level": "ground",
    },
    "Resources": {
        "atomic_mass": 2.0,
        "valence": 2,
        "electronegativity": 2.8,
        "preferred_state": "Solid",
        "energy_level": "ground",
    },
    "Access": {
        "atomic_mass": 3.0,
        "valence": 3,
        "electronegativity": 3.5,
        "preferred_state": "Liquid",
        "energy_level": "excited",
    },
    "Network": {
        "atomic_mass": 6.0,
        "valence": 6,
        "electronegativity": 4.0,
        "preferred_state": "Gas",
        "energy_level": "excited",
    },
    "Cyber": {
        "atomic_mass": 4.0,
        "valence": 4,
        "electronegativity": 3.8,
        "preferred_state": "Gas",
        "energy_level": "excited",
    },
    "Physical": {
        "atomic_mass": 3.5,
        "valence": 3,
        "electronegativity": 3.0,
        "preferred_state": "Solid",
        "energy_level": "ground",
    },
    "Execution": {
        "atomic_mass": 4.0,
        "valence": 4,
        "electronegativity": 3.5,
        "preferred_state": "Plasma",
        "energy_level": "ionized",
    },
    "Evasion": {
        "atomic_mass": 3.0,
        "valence": 4,
        "electronegativity": 2.7,
        "preferred_state": "Gas",
        "energy_level": "metastable",
    },
    "Transnational": {
        "atomic_mass": 7.0,
        "valence": 7,
        "electronegativity": 4.5,
        "preferred_state": "Plasma",
        "energy_level": "excited",
    },
}


def load_task_from_json(file_path: str) -> Optional[AdversaryTask]:
    """
    Load an adversary task from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        AdversaryTask object or None if loading fails
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle case-sensitivity issues in JSON keys
        # Check for ttPs vs ttps
        if "ttPs" in data and "ttps" not in data:
            data["ttps"] = data.pop("ttPs")

        # Set color based on category
        category = get_category_from_hash_id(data.get("hash_id", ""))
        data["color"] = TASK_COLORS.get(category, "#CCCCCC")

        # Set default atomic properties if not present
        if "atomic_properties" not in data:
            default_props = DEFAULT_ATOMIC_PROPERTIES.get(category, {})
            # Add reliability, confidence, maturity, complexity
            default_props.update(
                {
                    "reliability": 0.7,
                    "confidence": 0.7,
                    "maturity": 0.7,
                    "complexity": 0.5,
                }
            )
            data["atomic_properties"] = default_props

        # Add symbol if not present
        if "symbol" not in data:
            # Generate symbol from first letter of category
            data["symbol"] = category[0] if category else "X"

        # Add atomic number for periodic table
        if "atomic_number" not in data:
            # Use the SCH number as atomic number
            try:
                if data.get("hash_id", "").startswith("SCH"):
                    num_str = data["hash_id"].split("-")[0].replace("SCH", "")
                    data["atomic_number"] = int(num_str)
                else:
                    data["atomic_number"] = 0
            except:
                data["atomic_number"] = 0

        # Create the task object
        task = AdversaryTask.from_dict(data)
        logger.info(f"Loaded task {task.hash_id}: {task.task_name}")
        return task

    except Exception as e:
        logger.error(f"Error loading task from {file_path}: {str(e)}")
        return None


def load_tasks_from_directory(directory: str) -> List[AdversaryTask]:
    """
    Load all task JSON files from a directory.

    Args:
        directory: Directory path containing task JSON files

    Returns:
        List of AdversaryTask objects
    """
    tasks = []

    try:
        # Get all JSON files in the directory
        json_files = [
            f
            for f in os.listdir(directory)
            if f.endswith(".json") and "node_" in f.lower()
        ]

        for file_name in json_files:
            file_path = os.path.join(directory, file_name)
            task = load_task_from_json(file_path)
            if task:
                tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks from {directory}")
        return tasks

    except Exception as e:
        logger.error(
            f"Error loading tasks from directory {directory}: {str(e)}"
        )
        return []


def get_category_from_hash_id(hash_id: str) -> str:
    """
    Extract category from hash_id (e.g., SCH001-000 -> Planning)

    Args:
        hash_id: Hash ID in SCHxxx-xxx format

    Returns:
        Category name
    """
    if not hash_id or not hash_id.startswith("SCH"):
        return "Unknown"

    try:
        category_num = hash_id.split("-")[0].replace("SCH", "")
        categories = {
            "001": "Planning",
            "002": "Reconnaissance",
            "003": "Security",
            "004": "Resources",
            "005": "Access",
            "006": "Network",
            "007": "Cyber",
            "008": "Physical",
            "009": "Execution",
            "010": "Evasion",
            "011": "Transnational",
        }
        return categories.get(category_num, "Unknown")
    except:
        return "Unknown"


def create_demo_tasks() -> List[AdversaryTask]:
    """
    Create demo adversary tasks when no JSON files are available.

    Returns:
        List of demo AdversaryTask objects
    """
    tasks = []

    # Create 11 demo tasks (one for each category)
    for i in range(1, 12):
        category_num = f"{i:03d}"
        category = get_category_from_hash_id(f"SCH{category_num}-000")

        # Default properties
        atomic_props = DEFAULT_ATOMIC_PROPERTIES.get(category, {})
        atomic_props.update(
            {
                "reliability": 0.5 + (i * 0.04),
                "confidence": 0.6 + (i * 0.03),
                "maturity": 0.4 + (i * 0.05),
                "complexity": 0.3 + (i * 0.05),
            }
        )

        task = AdversaryTask(
            task_id=f"uuid-{category_num}-000-000",
            hash_id=f"SCH{category_num}-000",
            task_name=f"{category} Operations",
            description=f"Example task for demonstrating {category.lower()} operations in CTAS.",
            capabilities=f"This task demonstrates {category.lower()} capabilities.",
            limitations=f"This is a demo task with limited {category.lower()} context.",
            ttps=[
                f"Tactic: {category} Planning",
                f"Technique: {category} Execution",
                f"Procedure: {category} Analysis",
            ],
            relationships=f"This task connects to other tasks in the {category.lower()} workflow.",
            indicators=[
                f"Observable: {category} indicator 1",
                f"Observable: {category} indicator 2",
            ],
            historical_reference=f"This is a reference implementation of {category} tasks.",
            toolchain_refs=[
                f"tool_{category.lower()}_1",
                f"tool_{category.lower()}_2",
            ],
            eei_priority=[
                f"High: Critical {category.lower()} indicator",
                f"Medium: Important {category.lower()} behavior",
                f"Low: Background {category.lower()} activity",
            ],
            atomic_properties=atomic_props,
            color=TASK_COLORS.get(category, "#CCCCCC"),
        )

        tasks.append(task)

    logger.info(f"Created {len(tasks)} demo tasks")
    return tasks
