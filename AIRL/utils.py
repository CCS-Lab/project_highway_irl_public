import gym
import numpy as np

class StructEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.obs_a = self.env.reset()
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.obs_a = self.env.reset(**kwargs)
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode


class StructEnv_AIRL(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        #self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = self.env.reset()
        self.rew_episode = 0
        self.len_episode = 0
        self.rew_episode_airl = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = self.env.reset(**kwargs)
        self.rew_episode = 0
        self.rew_episode_airl = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        #self.observation_space.shape = (self.env.observation_space.shape[0] * self.env.observation_space.shape[1],)
        self.obs_a = self.env.reset(**kwargs)
        self.rew_episode = 0
        self.rew_episode_airl = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return ob, reward, done, info

    def step_airl(self, reward_airl):
        self.rew_episode_airl += reward_airl


    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_reward_airl(self):
        return self.rew_episode_airl

    def get_episode_length(self):
        return self.len_episode
    
class StructEnv_AIRL_Highway(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        #self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = self.env.reset()
        self.rew_episode = 0
        self.len_episode = 0
        self.rew_episode_airl = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = self.env.reset(**kwargs)
        self.rew_episode = 0
        self.rew_episode_airl = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        #self.observation_space.shape = (self.env.observation_space.shape[0] * self.env.observation_space.shape[1],)
        self.obs_a = self.env.reset(**kwargs)
        self.rew_episode = 0
        self.rew_episode_airl = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return ob, reward, done, info

    def step_airl(self, reward_airl):
        self.rew_episode_airl += reward_airl


    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_reward_airl(self):
        return self.rew_episode_airl

    def get_episode_length(self):
        return self.len_episode