# Based on https://gymnasium.farama.org/introduction/create_custom_env/
from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import utils
from distutils.util import strtobool
import pickle
import random
import json
from target_chords import TARGET_CHORD


class GridWorldEnv(gym.Env):

    def __init__(self, args, size):
        self.args                 = args

        self.observation_space    = gym.spaces.Box(low = -1, high = 100, shape=(600,), dtype = int)
        # Each # represents a specific chord (this mapping is automatically done when converting to a midi file.)
        self.CLASS_LIST = [43, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
        # Hard coded for speed:
        self.TARGET_CHORD = TARGET_CHORD

        self.idx_to_chord = {}
        # Get idx to chord mapping:
        for i in range(2, len(self.CLASS_LIST)):
            self.idx_to_chord[i] = self.CLASS_LIST[i]
        self.action_space         = gym.spaces.Discrete(len(self.CLASS_LIST) + 2) # For each of the most common chords + <end> + 'no chord'

        # Internal metrics
        self.steps                = 0
        self.cumulative_reward    = 0
        self.max_steps            = 600
        self.current_obs          = np.array([-1] * self.max_steps)
        self.all_locs              = []
        self.game_locs             = []
        self.all_traj              = []
        self.current_traj          = []
        self.current_acts          = []
        self.subob_traj            = []
        self.total_resets          = 0
        self.action_chords         = []
        self.prev_action           = -1

        self.composer_chords       = []
        self.num_iterations        = 100 # Number of times to train on a specific piece before switching
        self.curr_iterations       = 0

    
    def _get_obs(self):
        return np.array(self.TARGET_CHORD[:self.steps] + [-1] * (self.max_steps - self.steps))
        
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        print("Action chords:, ", self.action_chords)
        print(self.TARGET_CHORD)
        self.action_chords = []

        if len(self.composer_chords) == 0:
            with open('composer_chords.json', 'r') as file:
                composer_chords = json.load(file)
            assert self.args['composer'] in composer_chords.keys(), "Invalid composer, please see valid_composers.txt for valid composers"
            self.composer_chords = composer_chords[self.args['composer']]

        self.curr_iterations  += 1
        if self.curr_iterations > self.num_iterations:
            self.curr_iterations = 0
            self.TARGET_CHORD = random.choice(self.composer_chords)
        self.current_acts      = []
        self.current_traj      = []
        self.steps             = 0
        self.cumulative_reward = 0

        self.current_obs = [-1] * self.max_steps
        observation = self._get_obs()
        return observation, {}


    def step(self, action):
        reward = 0
        # We use `np.clip` to make sure we don't leave the grid bounds
        if action >= 0 and action <= 1:
            self.action_chords.append(int(action))
        else:
            self.action_chords.append(self.TARGET_CHORD[self.steps])
        if action >= 0 and action < len(self.CLASS_LIST):
            # action 0 = 'no chords'
            # action 1 = end current chord
            if int(action) == 0 or int(action) == 1:
                if int(action) == self.TARGET_CHORD[self.steps]:
                    reward += 1
            # Otherwise, assume guessing a chord:
            elif self.idx_to_chord[int(action)] == self.TARGET_CHORD[self.steps]:
                reward += 1
            else:
                reward = -1
        else: reward -= 1  
        if int(action) == self.prev_action and (self.prev_action == 1 or self.prev_action == 0):
            reward = -5
        # print(f'Target chord: {self.TARGET_CHORD[self.steps]}')
        self.steps             += 1
        observation             = self._get_obs()
        if self.steps > self.max_steps: terminated = True
        else: terminated = False
        self.cumulative_reward += reward
        self.current_traj.append(observation)
        truncated = False
        info = {"true path": self.TARGET_CHORD}
        self.prev_action = int(action)
        return observation, reward, terminated, truncated, info
    
    def save_trajectories(self):
        with open('traj.pkl', 'wb') as f:
            pickle.dump(self.all_traj, f)

    
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)