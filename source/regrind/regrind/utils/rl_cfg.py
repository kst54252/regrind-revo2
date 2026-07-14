from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class RslRlPpoCustomActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "CustomActorCritic"
    """The class name of the actor-critic policy."""

    zero_init_actor_last_layer: bool = False
    """If True, zero-initialize the actor's final linear layer so the Gaussian action mean is 0 at init."""
