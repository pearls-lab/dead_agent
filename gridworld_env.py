# Based on https://gymnasium.farama.org/introduction/create_custom_env/
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5, max_steps = 100):
        # The size of the square grid
        self.size                 = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location      = np.array([-1, -1], dtype=np.int32)
        self._target_location     = np.array([-1, -1], dtype=np.int32)
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "obs": gym.spaces.Box(0, size*2 + 1, dtype=float),
        #     }
        # )
        self.observation_space    = gym.spaces.Box(low = -99, high = 100, shape=(self.size**2 + 1,), dtype = float)
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space         = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),  # down
        }

        # Internal metrics
        self.steps                = 0
        self.cumulative_reward    = 0
        self.masterRewardDict     = {}
        self.tempRewardDict       = {}
        self.full_obs             = True
        self.event_locations      = []
        self.min_dist             = 5
        self.max_steps            = max_steps
        

    def get_grid(self):
        # Returns the current grid. Mainly used for value iteration and visualization
        grid                     = np.zeros((self.size, self.size))

        for event, details in self.masterRewardDict.items():
            location = details["location"]
            grid[location[0]][location[1]] = details["reward"]
        return grid
    
    def get_flattened_grid(self):
        return np.ravel(self.get_grid())

    def render(self):
        self.print_grid(self.get_grid())
        
    def print_grid(self, grid):
        longest_str = 10
        agent_loc   = (int(self._agent_location[0]), int(self._agent_location[1]))
        print(agent_loc)
        target_loc  = (int(self._target_location[0]), int(self._target_location[1]))
        print(target_loc)
        str_grid    = []
        # Iterate over columns
        for count, i in enumerate(grid):
            str_grid.append([])
            for count2, j in enumerate(i):
                if count == agent_loc[0] and count2 == agent_loc[1]: str_grid[count].append(self.pad_string("agent", longest_str))
                elif count == target_loc[0] and count2 == target_loc[1]: str_grid[count].append(self.pad_string("target", longest_str))
                else: str_grid[count].append(self.pad_string("", longest_str))

        for row in str_grid: print(row)
        
        return str_grid

    def pad_string(self, string, length):
        while len(string) < length: string += " "
        return string

    def _get_obs(self):
        if self.full_obs: return np.append(self.get_flattened_grid(), self._agent_location[0] + self.size * self._agent_location[1]) # Just appending agent location to end of array. not sure of better way of doing this
        return {"agent": self._agent_location, "target": self._target_location}
        
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "steps": self.steps,
            "totalReward" : self.cumulative_reward,
            "location": self._agent_location

        }
    
    def load_rewards(self, reward_dictionary):
        # Initializes master and temp reward dictionary
        # Temp is 'reloaded' from master when the env resets
        self.masterRewardDict = reward_dictionary.copy()
        self.tempRewardDict   = self.masterRewardDict.copy()
        self.event_locations  = [info["location"] for _, info in reward_dictionary.items()]
        if "goal" in reward_dictionary.keys(): self._target_location = np.array(reward_dictionary["goal"]["location"])
        if "start" in reward_dictionary.keys(): self._agent_location = np.array(reward_dictionary["start"]["location"])
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset Internal Metrics
        self.steps             = 0
        self.cumulative_reward = 0
        self.tempRewardDict    = self.masterRewardDict.copy()

        # Choose the agent's location uniformly at random if not specified
        if "start" not in self.masterRewardDict.keys(): self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else: self._agent_location = np.array(self.masterRewardDict["start"]["location"])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        if "goal" not in self.masterRewardDict.keys():
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def handle_events(self):
        # TODO: Do handling for special events or something
        reward_at_current_step = 0
        terminal               = False

        if not any((self._agent_location == subarray).all() for subarray in self.event_locations): return reward_at_current_step, terminal
        for event, info in self.masterRewardDict.items():
            if (self._agent_location == info["location"]).all(): 
                reward_at_current_step   += info['reward']
                info['available']         = False
                if not terminal: terminal = info['terminal'] # terminal event + non terminal event = terminal

        return reward_at_current_step, terminal

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[int(action)]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        self.steps += 1
        reward, terminated = self.handle_events()
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        if self.steps > self.max_steps: terminated = True
        return observation, reward, terminated, truncated, info
    
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)