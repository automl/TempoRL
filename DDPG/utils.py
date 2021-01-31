"""
The ReplayBuffer was originally implemented by Scott Fujimoto https://github.com/sfujim/TD3/blob/master/utils.py

We added a second replay buffer that can take repetitions/skips into account and created a version of the
Pendulum-v0 environment that has a different rendering style to display if actions were reactive or proactive
"""

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class FiGARReplayBuffer(object):
    def __init__(self, state_dim, action_dim, rep_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.rep = np.zeros((max_size, rep_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, rep, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.rep[self.ptr] = rep
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.rep[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


import gym


class Render(gym.Wrapper):
    """Render env by calling its render method.

    Args:
        env (gym.Env): Env to wrap.
        **kwargs: Keyword arguments passed to the render method.
    """

    def __init__(self, env, episode_modulo=1, **kwargs):
        super().__init__(env)
        self._kwargs = kwargs
        self.render_every_nth_episode = episode_modulo
        self._episode_counter = -1

    def reset(self, **kwargs):
        self._episode_counter += 1
        ret = self.env.reset(**kwargs)
        if self._episode_counter % self.render_every_nth_episode == 0:
            self.env.render(**self._kwargs)
        return ret

    def step(self, action):
        ret = self.env.step(action)
        if self._episode_counter % self.render_every_nth_episode == 0:
            self.env.render(**self._kwargs)
        return ret

    def close(self):
        self.env.close()


from gym.envs.classic_control import PendulumEnv
from os import path
import time
import inspect


class MyPendulum(PendulumEnv):

    def __init__(self, **kwargs):
        self.dec = False
        super().__init__(**kwargs)

    def is_decision_point(self):
        return self.dec

    def set_decision_point(self, b):
        self.dec = b

    def reset(self):
        self.dec = True
        return super().reset()

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            self.rod = rendering.make_capsule(1, .2)
            if self.is_decision_point():
                self.rod.set_color(.3, .3, .8)
                time.sleep(0.5)
                self.set_decision_point(False)
            else:
                self.rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            self.rod.add_attr(self.pole_transform)
            self.viewer.add_geom(self.rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(inspect.getfile(PendulumEnv)), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)

        if self.is_decision_point():
            self.rod.set_color(.3, .3, .8)
            # time.sleep(0.5)
            self.set_decision_point(False)
        else:
            self.rod.set_color(.8, .3, .3)

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


from gym.envs.registration import register

register(
    id='PendulumDecs-v0',
    entry_point='DDPG.utils:MyPendulum',
    max_episode_steps=200,
)
