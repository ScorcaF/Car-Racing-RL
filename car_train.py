import time

import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatNext, LaneKeepWrapper, ConcatObs, ReduceActionsWrapper, KeepCenterWrapper, GrayCropObservation
from stable_baselines3.common.monitor import Monitor


from stable_baselines3 import PPO
import os

models_dir = "PPO/graycrop_concat4_cnn"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
# steering, gas, and breaking.
#  Box([-1. 0. 0.], 1.0, (3,), float32)
# The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track.

env = Monitor(gym.make("CarRacing-v0"))
env = GrayCropObservation(env)
# env = LaneKeepWrapper(env, 2)
# env = KeepCenterWrapper(env, 10)
env = ConcatObs(env, 4)
# env = ReduceActionsWrapper(env)
# env = LaneForceWrapper(env)
# env.seed(1)
# env.reset()
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir, seed=0)

# Continue model training
model_path = models_dir + "/290000.zip"
model = PPO.load(model_path, env=env)

#Train
TIMESTEPS = 10_000
for i in range(30, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="graycrop_concat4_cnn")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# models_dir = "PPO/pred2steps_env"
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
# logdir = "logs"
# if not os.path.exists(logdir):
#     os.makedirs(logdir)
#
# wrapped_env = ConcatNext(env)
# # aigym_path = "Recs"
# # wrapped_env = gym.wrappers.Monitor(wrapped_env, aigym_path, video_callable=False, force=True)
#
# # Training
# obs = wrapped_env.reset()
# model = PPO('MlpPolicy', wrapped_env, verbose=1, tensorboard_log=logdir)
# TIMESTEPS = 10_000
# for i in range(10,20):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="pred2steps_env")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")