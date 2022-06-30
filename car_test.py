import time

import gym
from collections import deque
from gym import spaces
import numpy as np
from Utils import RewardWrapper, ConcatActs


from stable_baselines3 import PPO
import os


env = gym.make("CarRacing-v0")
# env = RewardWrapper(ConcatActs(env, 2))

models_dir = "PPO/baseline_env"
model_path = models_dir + "/40000.zip"
model = PPO.load(model_path, env=env)


# Visualize results
episodes = 10

for _ in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs.copy(), deterministic=True)
        obs, rewards, done, info = env.step(action)

        env.render()
        # time.sleep(0.01)
# 		print(rewards)

env.close()


# ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
