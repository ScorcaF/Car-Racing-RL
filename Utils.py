import gym
from collections import deque
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import cv2




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
        pred_acc_reward = -0.1 * np.linalg.norm(self.pres_action - self.pred_action)
        reward = driving_reward + pred_acc_reward
        self.pred_action = action[3:]
        return ob, reward, done, info



# Not used
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        reward += self.penalty
        return reward


# class TensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """
#     def __init__(self, verbose=0):
#         self.is_tb_set = False
#         super(TensorboardCallback, self).__init__(verbose)
#
#     def _on_step(self) -> bool:
#         # Log additional tensor
#         if not self.is_tb_set:
#             with self.model.graph.as_default():
#                 tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
#                 self.model.summary = tf.summary.merge_all()
#             self.is_tb_set = True
#         # Log scalar value (here a random variable)
#         value = np.random.random()
#         summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
#         self.locals['writer'].add_summary(summary, self.num_timesteps)
#         return True



    # # horizontal cordinates of center of the road in the cropped slice
    # mid = 24
    #
    # # some further adjustments obtained through trail and error
    # if nz[:, 0, 0].max() == nz[:, 0, 0].min():
    #     if nz[:, 0, 0].max() < 30 and nz[:, 0, 0].max() > 20:
    #         return previous_error
    #     if nz[:, 0, 0].max() >= mid:
    #         return (-15)
    #     else:
    #         return (+15)
    # else:
    #     return (((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid)
    # return (green)

class LaneKeepWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(2,49), dtype=env.observation_space.dtype)


    def reset(self):
        ob = self.env.reset()
        return self.process_obs(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.process_obs(ob)
        out_lane_reward = self.lane_keep_reward(ob)
        reward += out_lane_reward
        return ob, reward, done, info

    def lane_keep_reward(self, obs):
        # find all non zero values in the cropped strip.
        # These non zero points(white pixels) corresponds to the edges of the road
        nz = cv2.findNonZero(obs)

        # center of the image: 24
        if nz is None:
            return -10
        elif nz[:, 0, 0].max() < 24:
            return -0.2*(nz[:, 0, 0].max() - 24)
        elif nz[:, 0, 0].min() > 24:
            return -0.2*(24 - nz[:, 0, 0].min())
        else:
            return 0.005*(nz[:, 0, 0].max() - 24) + 0.005*(24 - nz[:, 0, 0].min())

    def process_obs(self, obs):
        # Apply following functions
        obs = self.green_mask(obs)
        obs = self.gray_scale(obs)
        obs = self.blur_image(obs)
        obs = self.crop(obs)
        obs = self.canny_edge_detector(obs)


        return obs

    def green_mask(self, observation):
        hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

        ## slice the green
        imask_green = mask_green > 0
        green = np.zeros_like(observation, np.uint8)
        green[imask_green] = observation[imask_green]
        return (green)

    def gray_scale(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return gray

    def blur_image(self, observation):
        blur = cv2.GaussianBlur(observation, (5, 5), 0)
        return blur

    def canny_edge_detector(self, observation):
        canny = cv2.Canny(observation, 50, 150)
        return canny

    def crop(self, observation):
        return observation[63:65, 24:73]