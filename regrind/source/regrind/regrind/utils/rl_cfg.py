from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlZeroInitMLPModelCfg(RslRlMLPModelCfg):
    """Configuration for an RSL-RL 5 MLP with a zero-initialized output layer."""

    class_name: str = "regrind.modules.actor_critic:ZeroInitMLPModel"
