import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from typing import Any, Union, Optional, Dict, List, ClassVar
import enlight.utils as U
import enlight.rl.actor

import hydra


class SAC(object):
    def __init__(
            self, *,
            actor: nn.Module,
            actor_update_freq: int = 2,
            actor_lr: Optional[float] = None,
            actor_optimizer: Optional[torch.optim.Optimizer] = None,
            action_range,
            action_shape,
            critic: nn.Module,
            critic_lr: Optional[float] = None,
            critic_optimizer: Optional[torch.optim.Optimizer] = None,
            critic_tau: float,
            critic_target_update_freq: int = 2,
            init_temperature: float,
            discount: float = 0.99,
            batch_size: int,
            log_alpha_lr: Optional[float] = None,
            log_alpha_optimizer: Optional[torch.optim.Optimizer] = None,
            target_entropy: Union[float, str] = 'auto',
            device: str = 'cuda',
    ):
        """
        Args:
            actor_optimizer: if specified, LR will be ignored
        """
        self.device = device
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = U.clone_model(self.critic)

        # tie conv layers between actor and critic
        # TODO: address encoder
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        # optimizers
        self.actor_optimizer = self._create_optimizer(
            self.actor.parameters(), actor_lr, actor_optimizer
        )
        self.critic_optimizer = self._create_optimizer(
            self.critic.parameters(), critic_lr, critic_optimizer
        )
        self.log_alpha_optimizer = self._create_optimizer(
            [self.log_alpha], log_alpha_lr, log_alpha_optimizer
        )

        if target_entropy == 'auto':
            # heuristic: set target entropy to -|A|
            self.target_entropy = -np.prod(action_shape)
        else:
            self.target_entropy = float(target_entropy)

        self.discount = discount
        self.batch_size = batch_size
        self.actor_update_freq = actor_update_freq
        self.action_range = action_range
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def _create_optimizer(self, params, lr, optimizer):
        if lr is None and optimizer is None:
            raise ValueError('must specify either lr or torch.optim.Optimizer. '
                             'If the latter is specified, lr will be ignored.')
        if optimizer is not None:
            return optimizer
        else:
            return torch.optim.Adam(params, lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        return enlight.rl.actor.act(
            actor=self.actor,
            obs=obs,
            sample=sample,
            device=self.device,
            action_range=self.action_range
        )

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_freq == 0:
            U.update_soft_params(
                self.critic, self.critic_target, self.critic_tau
            )
