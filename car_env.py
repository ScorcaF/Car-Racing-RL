import time

import gym
from collections import deque
from gym import spaces
import numpy as np

from stable_baselines3 import PPO
import os

#CLASSES DEFINITION ####################################################################################################
class ConcatObs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.array(self.frames)

class ConcatActs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.actions = deque([], maxlen=k)
        shp = env.action_space.shape
        self.action_space = spaces.Box(low=np.tile(np.array([-1., 0., 0.]), (k,1)), high=np.tile(np.array([1., 1., 1.]), (k, 1)), shape=((k,) + shp), dtype=env.action_space.dtype)

    def reset(self):
        ob = self.env.reset()
        self.last_action = None
        for _ in range(self.k):
            self.actions.append(np.empty(self.action_space.shape))
        return ob

    def step(self, action):
        self.last_action = action[0]
        self.action = action
        ob, reward, done, info = self.env.step(self.last_action)
        self.actions.append(self.last_action)
        return ob, reward, done, info


# class RewardWrapper(gym.RewardWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def reward(self, reward):
#         reward += np.linalg.norm(self.actions - self.action)
#         return reward

#CLASSES DEFINITION ####################################################################################################

# steering, gas, and breaking.
#  Box([-1. 0. 0.], 1.0, (3,), float32)
# The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track.
# Create the environment
env = gym.make("CarRacing-v0")
# env.reset()

wrapped_env = ConcatActs(env=ConcatObs(env, 4), k=3)


# Reset the Env
obs = wrapped_env.reset()
for _ in range(2):
    obs, _, _, _  = wrapped_env.step(wrapped_env.action_space.sample())
    wrapped_env.render()
    print(np.array(wrapped_env.actions).shape, wrapped_env.action.shape)
    time.sleep(0.01)



