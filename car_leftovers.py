import time
import matplotlib.pyplot as plt
import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatNext, LaneKeepWrapper, ConcatObs, KeepCenterWrapper, NormalizeObservation, TensorboardCallback
from PIL import Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from stable_baselines3 import PPO
import os

env = Monitor(gym.make("CarRacing-v0"))
# env = ConcatNext(env)
env = LaneKeepWrapper(env, 1/1000)
# env = LaneForceWrapper(env)
env.reset()


# Environment setup sanity checks
# obs = env.reset()
# for _ in range(1000):
#     obs, rewards, done, info = env.step(env.action_space.sample())
#     env.render()
#     print(rewards)



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



video_folder = 'logs/videos/'
video_length = 100
env_id = "CarRacing-v0"
env = DummyVecEnv([lambda: gym.make(env_id)])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent-{env_id}")

env.reset()
for _ in range(video_length + 1):
  action = [env.action_space.sample()]
  obs, _, _, _ = env.step(action)
# Save the video
# env.close()