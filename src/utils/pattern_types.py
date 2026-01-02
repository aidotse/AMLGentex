"""
Centralized pattern type ID mappings for AML simulation.

This module defines the standard IDs for different pattern types used throughout
the data generation, preprocessing, visualization, and ML pipeline.

Pattern types are categorized into:
- SAR patterns (IDs 1-8): Suspicious activity patterns
- Normal patterns (IDs 10-15): Legitimate transaction patterns
- Generic (ID 0): Non-pattern account behaviors (income, spending, etc.)

The patternID field in transactions identifies specific pattern instances,
while the modelType field identifies the pattern type category.
"""

# Generic account behaviors (not specific patterns)
GENERIC_BEHAVIOR_TYPE = 0  # Income, expenses, spending, etc.

# SAR (Suspicious Activity Report) pattern types
# These represent different money laundering typologies
SAR_PATTERN_TYPES = {
    "fan_out": 1,      # One account sends to many
    "fan_in": 2,       # Many accounts send to one
    "cycle": 3,        # Circular transaction chain
    "bipartite": 4,    # Two-layer transaction structure
    "stack": 5,        # Layered money movement
    "random": 6,       # Random suspicious pattern
    "scatter_gather": 7,  # Scatter then gather funds
    "gather_scatter": 8   # Gather then scatter funds
}

# Normal (legitimate) transaction pattern types
# These represent typical banking behaviors
NORMAL_PATTERN_TYPES = {
    "single": 10,      # One-time transaction
    "fan_out": 11,     # Legitimate one-to-many (e.g., payroll)
    "fan_in": 12,      # Legitimate many-to-one (e.g., collections)
    "forward": 13,     # Sequential forwarding
    "mutual": 14,      # Mutual exchanges between accounts
    "periodical": 15   # Regular scheduled transactions
}

# Reverse mappings (int -> str)
SAR_ID_TO_NAME = {v: k for k, v in SAR_PATTERN_TYPES.items()}
NORMAL_ID_TO_NAME = {v: k for k, v in NORMAL_PATTERN_TYPES.items()}

# Combined mappings for convenience
ALL_PATTERN_TYPES = {**SAR_PATTERN_TYPES, **NORMAL_PATTERN_TYPES}
ALL_ID_TO_NAME = {
    **SAR_ID_TO_NAME,
    **NORMAL_ID_TO_NAME,
    GENERIC_BEHAVIOR_TYPE: "generic"
}

def get_pattern_name(type_id: int) -> str:
    """
    Get pattern name from type ID.

    Args:
        type_id: Pattern type ID (1-8 for SAR, 10-15 for normal, 0 for generic)

    Returns:
        Pattern name string
    """
    return ALL_ID_TO_NAME.get(type_id, f"unknown_{type_id}")

def get_pattern_id(pattern_name: str) -> int:
    """
    Get pattern type ID from name.

    Args:
        pattern_name: Pattern name string

    Returns:
        Pattern type ID, or None if not found
    """
    return ALL_PATTERN_TYPES.get(pattern_name, None)

def is_sar_pattern(type_id: int) -> bool:
    """Check if type ID corresponds to a SAR pattern (1-8)."""
    return 1 <= type_id <= 8

def is_normal_pattern(type_id: int) -> bool:
    """Check if type ID corresponds to a normal pattern (10-15)."""
    return 10 <= type_id <= 15

def is_generic(type_id: int) -> bool:
    """Check if type ID corresponds to generic account behavior (0)."""
    return type_id == 0
