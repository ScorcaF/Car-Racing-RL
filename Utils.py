import gym
from collections import deque
from gym import spaces
import numpy as np


# Not used
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

# assert len(action_space.shape) == 1, "Error: the action space must be a vector"
# AssertionError: Error: the action space must be a vector
#To extend to k action predictions
# class ConcatActs(gym.Wrapper):
#     def __init__(self, env, k):
#         gym.Wrapper.__init__(self, env)
#         self.k = k
#         self.actions = deque([], maxlen=k)
#         shp = env.action_space.shape
#         self.action_space = spaces.Box(low=np.tile(np.array([-1., 0., 0.]), (k,1)), high=np.tile(np.array([1., 1., 1.]), (k, 1)), shape=((k,) + shp), dtype=env.action_space.dtype)
#
#     def reset(self):
#         ob = self.env.reset()
#         self.last_action = None
#         for _ in range(self.k):
#             self.actions.append(np.empty(3,))
#         return ob
#
#     def step(self, action):
#         self.last_action = action[0]
#         self.action = action
#         ob, reward, done, info = self.env.step(self.last_action)
#         self.actions.append(self.last_action)
#         return ob, reward, done, info


class ConcatNext(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # Issue: initialization will mislead rewards in first step
        self.pres_action = np.empty((3,))
        self.pred_action = np.empty((3,))
        self.action_space = spaces.Box(low=np.tile(np.array([-1., 0., 0.]), 2),
                                       high=np.tile(np.array([1., 1., 1.]), 2), shape=(6,), dtype=env.action_space.dtype)

    # Error: int() argument must be a string, a bytes-like object or a number, not 'NoneType'
    # def reset(self):
    #     #Issue: initialization will mislead rewards in first step
    #     ob = self.env.reset()
    #     self.pres_action = np.empty((3,))
    #     self.pred_action = np.empty((3,))


    def step(self, action):
        self.pres_action = action[:3]
        ob, driving_reward, done, info = self.env.step(self.pres_action)
        pred_acc_reward = 0.1 * np.linalg.norm(self.pres_action - self.pred_action)
        reward = driving_reward + pred_acc_reward
        self.pred_action = action[3:]
        return ob, reward, done, info



# Not used
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        print(reward)
        reward -= 0.1*np.linalg.norm(self.last_action - self.pred_action)
        print("Old prediction: ", self.pred_action)
        print(reward)
        return reward