"""Gym registration for the RB3-730 + Revo2 + tuna can REGRIND MDP task."""

import gymnasium as gym

from . import agents


_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:RB3Revo2TunaPPORunnerCfg"
_TRANSFER_AGENT_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:RB3Revo2TunaTransferPPORunnerCfg"
)


def _register(task_id: str, env_cfg_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rb3_revo2_tuna_env_cfg:{env_cfg_name}",
            "rsl_rl_cfg_entry_point": _AGENT_CFG,
        },
    )


_register("Regrind-RB3-Revo2-TunaCan-v0", "RB3Revo2TunaEnvCfg")
_register("Regrind-RB3-Revo2-TunaCan-Smoke-v0", "RB3Revo2TunaEnvCfg_SMOKE")
_register("Regrind-RB3-Revo2-TunaCan-Play-v0", "RB3Revo2TunaEnvCfg_PLAY")

# The online bridge loads a floating-hand checkpoint unchanged (67-D actor
# observation, 12-D action), solves the wrist action with bounded RB3 IK, and
# drives Revo2's six leaders directly in the assembled physics scene.
for task_id, env_cfg_name in (
    ("Regrind-RB3-Revo2-TunaCan-Online-v0", "RB3Revo2TunaOnlineEnvCfg"),
    (
        "Regrind-RB3-Revo2-TunaCan-Online-Smoke-v0",
        "RB3Revo2TunaOnlineEnvCfg_SMOKE",
    ),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.rb3_revo2_online_env_cfg:{env_cfg_name}"
            ),
            "rsl_rl_cfg_entry_point": _TRANSFER_AGENT_CFG,
        },
    )

gym.register(
    id="Regrind-RB3-Revo2-TunaCan-Online-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rb3_revo2_online_env_cfg:RB3Revo2TunaOnlineEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _AGENT_CFG,
    },
)

# Backward-compatible names used by the earlier validation scripts.
_register("Regrind-RB3-Revo2-Tuna-v0", "RB3Revo2TunaEnvCfg")
_register("Regrind-RB3-Revo2-Tuna-Play-v0", "RB3Revo2TunaEnvCfg_PLAY")
