import time

import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatNext, LaneKeepWrapper, ConcatObs, ReduceActionsWrapper, KeepCenterWrapper, GrayCropObservation, ForceCenterWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv



from stable_baselines3 import PPO
import os

import base64
from pathlib import Path


env = gym.make("CarRacing-v0")
# env = KeepCenterWrapper(env, 10, manipulate_obs=False)
env = ReduceActionsWrapper(env)
# env = ConcatObs(env,10)
# env = RewardWrapper(env)
# env = LaneForceWrapper(env)
env = GrayCropObservation(env, gray=False)
# env = ConcatObs(env, 4)


models_dir = "PPO/crop_actpenalty_cnn"
model_path = models_dir + "/420000.zip"
model = PPO.load(model_path, env=env)


# Visualize results
# episodes = 10
#
# for _ in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs.copy(), deterministic=True)
#         obs, rewards, done, info = env.step(action)
#
#         env.render()
#         time.sleep(0.01)
#
# env.close()

video_folder = 'videos/'
video_length = 2000
env_id = "CarRacing-v0"
env = DummyVecEnv([lambda: env])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"crop_actpenalty_cnn33")

env.reset()
for _ in range(video_length + 1):
    action, _states = model.predict(obs.copy(), deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Save the video
env.close()






