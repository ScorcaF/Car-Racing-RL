import time
import matplotlib.pyplot as plt
import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatNext, LaneKeepWrapper, ConcatObs, KeepCenterWrapper, NormalizeObservation
from PIL import Image


from stable_baselines3 import PPO
import os

env = gym.make("CarRacing-v0")
# env = ConcatNext(env)
env = KeepCenterWrapper(env, 10)
# env = LaneForceWrapper(env)
env.reset()


# Environment setup sanity checks
obs = env.reset()
for _ in range(1000):
    obs, rewards, done, info = env.step(env.action_space.sample())
    env.render()
    print(rewards)



# print(obs.shape)
# env_screen = env.render(mode = 'rgb_array')
# env.close()
# Image.fromarray(env_screen).save(f"reset_screen.png")

# Training sanity checks
# obs = env.reset()
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="logs")
# TIMESTEPS = 2
# for i in range(2):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="tensorboard_test", callback=[TensorboardCallback()])



