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

from IPython import display as ipythondisplay

env = gym.make("CarRacing-v0")
env = KeepCenterWrapper(env, 10)
# env = ReduceActionsWrapper(env)
# env = ConcatObs(env,3)
# env = RewardWrapper(env)
# env = LaneForceWrapper(env)


models_dir = "PPO/hard_center_penalty_env"
model_path = models_dir + "/290000.zip"
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
        print(rewards)

env.close()

#Record video
# # Set up fake display; otherwise rendering will fail
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'



# def show_videos(video_path='', prefix=''):
#   """
#   Taken from https://github.com/eleurent/highway-env
#
#   :param video_path: (str) Path to the folder containing videos
#   :param prefix: (str) Filter the video, showing only the only starting with this prefix
#   """
#   html = []
#   for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
#       video_b64 = base64.b64encode(mp4.read_bytes())
#       html.append('''<video alt="{}" autoplay
#                     loop controls style="height: 400px;">
#                     <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                 </video>'''.format(mp4, video_b64.decode('ascii')))
#   ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
#
# def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
#   """
#   :param env_id: (str)
#   :param model: (RL model)
#   :param video_length: (int)
#   :param prefix: (str)
#   :param video_folder: (str)
#   """
#   eval_env = DummyVecEnv([lambda: env_id])
#   # Start the video at step=0 and record 500 steps
#   eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
#                               record_video_trigger=lambda step: step == 0, video_length=video_length,
#                               name_prefix=prefix)
#
#   obs = eval_env.reset()
#   for _ in range(video_length):
#     action, _ = model.predict(obs)
#     obs, _, _, _ = eval_env.step(action)
#
#   # Close the video recorder
#   eval_env.close()
#
# record_video(env, model, video_length=500, prefix='baseline')
# show_videos('videos', prefix='baseline')


