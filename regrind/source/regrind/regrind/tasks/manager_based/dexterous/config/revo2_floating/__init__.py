"""Gym registration for floating Revo2 + tuna-can REGRIND tasks."""

import gymnasium as gym

from . import agents


_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:FloatingRevo2TunaPPORunnerCfg"


def _register(task_id: str, env_cfg_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.revo2_floating_tuna_env_cfg:{env_cfg_name}",
            "rsl_rl_cfg_entry_point": _AGENT_CFG,
        },
    )


_register("Regrind-Floating-Revo2-TunaCan-v0", "FloatingRevo2TunaEnvCfg")
_register("Regrind-Floating-Revo2-TunaCan-Smoke-v0", "FloatingRevo2TunaEnvCfg_SMOKE")
_register("Regrind-Floating-Revo2-TunaCan-Play-v0", "FloatingRevo2TunaEnvCfg_PLAY")
