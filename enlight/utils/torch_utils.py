import numpy as np
import torch
import random
import torch.nn as nn
from copy import deepcopy


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


# ========== module operations =========
def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_params(model):
    set_requires_grad(model, False)


def unfreeze_params(model):
    set_requires_grad(model, True)


def clip_grad_value(model, max_value):
    with torch.no_grad():
        nn.utils.clip_grad_value_(model.parameters(), max_value)


def clip_grad_norm(model, max_norm, norm_type=2):
    """
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    with torch.no_grad():
        return nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type
        )


def count_parameters(model):
    return sum(x.numel() for x in model.parameters())


def get_model_device(model):
    """
    Returns:
        first model parameter's device
    """
    return next(model.parameters()).device


def clone_model(model):
    # with torch.no_grad():
    new_model = deepcopy(model).to(get_model_device(model))
    print('clone no grad')
        # new_model.load_state_dict(model.state_dict())
    return new_model


def update_soft_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
