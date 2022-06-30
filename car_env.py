import time

import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatNext


from stable_baselines3 import PPO
import os

models_dir = "PPO/baseline_env"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = gym.make("CarRacing-v0")

# steering, gas, and breaking.
#  Box([-1. 0. 0.], 1.0, (3,), float32)
# The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track.
# env.reset()
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# TIMESTEPS = 10_000
# for i in range(30):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="baseline_env")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")

aigym_path = "Recs"
models_dir = "PPO/pred2steps_env"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

wrapped_env = ConcatNext(env)
# wrapped_env = gym.wrappers.Monitor(wrapped_env, aigym_path, video_callable=False, force=True)

# Training
obs = wrapped_env.reset()
model = PPO('MlpPolicy', wrapped_env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 10_000
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="pred2steps_env")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# Sanity checks
# obs = wrapped_env.reset()
# for _ in range(2):
#     obs, rewards, done, info = wrapped_env.step(wrapped_env.action_space.sample())
    # print("Historical actions: ", np.array(wrapped_env.actions).reshape(1, -1).squeeze().shape)
    # print("Actions and Predictions: ", wrapped_env.action.shape)

# obs = wrapped_env.reset()
# model = PPO('MlpPolicy', wrapped_env, verbose=1)
# TIMESTEPS = 1000
# for i in range(2):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#     model.save(f"{models_dir}/{TIMESTEPS*i}")

