import torch
import torch.nn as nn
import enlight.utils as U


__all__ = ['act']


def act(
        actor: nn.Module,
        obs,
        action_range,
        sample=False,
        device=None
):
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    obs = obs.unsqueeze(0)
    dist = actor(obs)
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*action_range)
    assert action.ndim == 2 and action.shape[0] == 1
    return U.to_np(action[0])
