import time

import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatNext, LaneKeepWrapper, ConcatObs, ReduceActionsWrapper, KeepCenterWrapper, NormalizeObservation, ForceCenterWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv



from stable_baselines3 import PPO
import os

import base64
from pathlib import Path


env = gym.make("CarRacing-v0")
env = KeepCenterWrapper(env, 10)
# env = ReduceActionsWrapper(env)
# env = ConcatObs(env,3)
# env = RewardWrapper(env)
# env = LaneForceWrapper(env)


models_dir = "PPO/lanekeep_div2_env"
model_path = models_dir + "/500000.zip"
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
#         # time.sleep(0.01)
#         print(rewards)
#
# env.close()

video_folder = 'videos/'
video_length = 5000
env_id = "CarRacing-v0"
env = DummyVecEnv([lambda: env])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"lanekeep_div2_env")

env.reset()
for _ in range(video_length + 1):
    action, _states = model.predict(obs.copy(), deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Save the video
env.close()






