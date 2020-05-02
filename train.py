import os
import sys
import time
import numpy as np

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from enlight.envs.dm_control import make_visual_benchmark
from enlight.utils.tracer import Tracer
import enlight.utils as U
from enlight.rl.algo import SAC

# TODO clean up these imports
from enlight.utils.logger import Logger
from enlight.utils.video import VideoRecorder
# TODO use tianshou replay instead
from enlight.rl.replay_buffer import ReplayBuffer

torch.backends.cudnn.benchmark = True


def apply_debug_settings(cfg):
    cfg.exp_dir = 'debug'
    cfg.num_seed_steps = 3
    cfg.num_train_steps = 60
    cfg.replay_buffer_capacity = 12
    cfg.device = 'cpu'
    cfg.lr = 0.5
    cfg.batch_size = 8
    agent = cfg.agent.params
    agent.actor_update_freq = 2
    agent.critic_target_update_freq = 2
    agent.critic_tau = 0.3
    agent.discount = 0.8
    agent.init_temperature = 0.1
    cfg.actor.params.hidden_dim = cfg.critic.params.hidden_dim = 64


class Workspace(object):
    def __init__(self, cfg, tracer):
        self.tracer = tracer

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        U.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_visual_benchmark(
            cfg.env,
            seed=cfg.seed,
            image_height=cfg.image_size,
            image_width=cfg.image_size,
            action_repeat=cfg.action_repeat,
            frame_stack=cfg.frame_stack
        )

        cfg.task.obs_shape = self.env.observation_space.shape
        cfg.task.action_shape = self.env.action_space.shape
        cfg.task.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        actor = hydra.utils.instantiate(cfg.actor)
        critic = hydra.utils.instantiate(cfg.critic)

        agent_params = OmegaConf.to_container(cfg.agent.params, resolve=True)
        # agent_params['cfg'] = cfg   # TODO debugging only
        agent_params['actor'] = actor
        agent_params['critic'] = critic

        self.agent = SAC(**agent_params)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        tracer = self.tracer

        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with U.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

                tracer.event('eval_action', action)
                tracer.event('eval_obs', obs)
                tracer.event('eval_reward', reward)
                if tracer.enabled and episode_step > 10:
                    break

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        tracer = self.tracer

        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with U.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)
            tracer.event('action', action)
            tracer.event('next_obs', next_obs)
            tracer.event('reward', reward)
            tracer.event('done', done)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

            tracer.event('actor', self.agent.actor)
            tracer.event('critic', self.agent.critic)
            tracer.event('critic_target', self.agent.critic_target)
            tracer.event('log_alpha', self.agent.log_alpha)
        tracer.done()


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W

    tracer = Tracer(
        f'~/log/drq/cheetah_run_seed={cfg.seed}.trace',
        mode=cfg.debug.trace_mode,
        verbose=cfg.debug.verbose
    )
    if tracer.enabled:
        apply_debug_settings(cfg)

    workspace = W(cfg, tracer)
    workspace.run()


if __name__ == '__main__':
    main()
