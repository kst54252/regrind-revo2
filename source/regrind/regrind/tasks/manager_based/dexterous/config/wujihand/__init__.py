# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Regrind-WujiHand-Scissors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_scissors_env_cfg:WujiHandScissorsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScissorsPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Scissors-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_scissors_env_cfg:WujiHandScissorsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScissorsPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Screwdriver-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_screwdriver_env_cfg:WujiHandScrewdriverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScrewdriverPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Screwdriver-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_screwdriver_env_cfg:WujiHandScrewdriverEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScrewdriverPPORunnerCfg",
    },
)
