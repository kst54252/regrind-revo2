"""Forward existing observation/history output to a frozen policy without remapping."""
import torch


class FrozenPolicyAdapter:
    def __init__(self,policy):
        self.policy=policy.eval()
        self.policy.requires_grad_(False)
        self.initial={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}

    def __call__(self,observations):
        # ObservationManager already owns term order, histories and reset.
        # The policy owns its checkpoint normalizer. Do not normalize twice.
        actor=observations['policy']
        if actor.shape!=(1,67) or not torch.isfinite(actor).all():raise ValueError('Expected finite actor (1,67)')
        with torch.inference_mode():
            action=self.policy(observations)
        if action.shape!=(1,12) or not torch.isfinite(action).all():raise ValueError('Expected finite action (1,12)')
        return action

    def assert_frozen(self):
        for k,v in self.policy.state_dict().items():
            if not torch.equal(v.detach().cpu(),self.initial[k]):raise AssertionError('Policy/normalizer changed: '+k)
