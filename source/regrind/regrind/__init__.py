# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

try:
    # Register Gym environments (requires Isaac Sim runtime).
    from .tasks import *

except (ImportError, ModuleNotFoundError) as e:
    import warnings

    warnings.warn(
        f"regrind: Isaac Sim / omni not available — {e}. Sim-free submodules (data, modules) are still importable.",
        stacklevel=2,
    )

