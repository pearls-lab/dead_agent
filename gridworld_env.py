# Based on https://gymnasium.farama.org/introduction/create_custom_env/
from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5, max_steps: int = 100):
        # The size of the square grid
        self.size                 = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location      = np.array([-1, -1], dtype=np.int32)
        self._target_location     = np.array([-1, -1], dtype=np.int32)
        self._agent_identifier    = -99
        self._goal_identifier     = 99
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "obs": gym.spaces.Box(0, size*2 + 1, dtype=float),
        #     }
        # )
        self.observation_space    = gym.spaces.Box(low = -99, high = 100, shape=(self.size**2,), dtype = float)
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
        self.locIdenDict          = []
        self.allIdentifiers       = [self._agent_identifier, self._goal_identifier]
        for _, value in self.masterRewardDict.items():
            info       = value
            identifier = np.random.randint(1, 100)
            while identifier in self.allIdentifiers: identifier = np.random.randint(1, 100)
            self.allIdentifiers.append(identifier)
            for location in info['location']: self.locIdenDict[location] = identifier
        

    def print_target_agent(self):
        print("Agent:", self._agent_location)
        print("Target:", self._target_location)

    def get_grid(self):
        grid                                                     = np.zeros((self.size, self.size))
        grid[self._agent_location[0]][self._agent_location[1]]   = self._agent_identifier
        for loc, iden in self.locIdenDict: grid[loc[0]][loc[1]]  = iden
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
    
    def print_path(self, path, return_string = False):
        # Path should be an array of locations
        grid = self.get_grid()
        longest_str = 10
        agent_loc   = (int(self._agent_location[0]), int(self._agent_location[1]))
        print(agent_loc)
        target_loc  = (int(self._target_location[0]), int(self._target_location[1]))
        print(target_loc)
        str_grid    = []
        if 'start' in self.masterRewardDict.keys(): start_loc = self.masterRewardDict['start']['location'][0]
        # Iterate over columns
        for count, i in enumerate(grid):
            str_grid.append([])
            for count2, j in enumerate(i):
                if count == agent_loc[0] and count2 == agent_loc[1]: str_grid[count].append(self.pad_string("agent", longest_str))
                elif count == target_loc[0] and count2 == target_loc[1]: str_grid[count].append(self.pad_string("target", longest_str))
                elif count * 10 + count2 in path: str_grid[count].append(self.pad_string("explored", longest_str))
                elif count == start_loc[0] and count2 == start_loc[1]: str_grid[count].append(self.pad_string("start", longest_str))
                else: str_grid[count].append(self.pad_string("", longest_str))

        for row in str_grid: print(row)
        
        if return_string: 
            final_str = ""

            for row in str_grid:
                for col in str_grid:
                    final_str += str(col)
                final_str += "\n"
            return final_str
        return str_grid

    def print_path_image(self, path, save_path, title):
        cmap           = plt.get_cmap('Greens', 100)
        norm           = plt.Normalize(0, self.size)
        rgba           = cmap(norm(self.get_grid()))
        colored_spots  = []

        if 'start' in self.masterRewardDict.keys(): location = self.masterRewardDict['start']['location'][0]
        else:                                       location = self._agent_location

        # Iniitalize start square
        # flattened_location       = 10*location[0] + location[1]
        # rgba[flattened_location] = 0.0, 0.5, 0.8, 1.0
        # colored_spots.append(flattened_location)
        rgba[location[0]][location[1]] = 0.0, 0.5, 0.8, 1.0
        colored_spots.append(location)


        if 'goal' in self.masterRewardDict.keys(): location = self.masterRewardDict['goal']['location'][0]
        else:                                      location = self._target_location

        # Initialize goal location
        # flattened_location       = 10*location[0] + location[1]
        # rgba[flattened_location] = 0.8, 0.5, 0.0, 1.0
        # colored_spots.append(flattened_location)
        rgba[location[0]][location[1]] = 0.8, 0.5, 0.0, 1.0
        colored_spots.append(location)

        # Highlight path taken
        fig, ax = plt.subplots()
        im      = ax.imshow(rgba, interpolation='nearest')
        for i in range(self.size):
            for j in range(self.size):
                if (i * 10 + j) in path: ax.text(j, i, "v", ha="center", va="center", color="r", bbox=dict(facecolor='none', edgecolor='red'))
                else: ax.text(j, i, ".", ha="center", va="center", color="g", bbox=dict(facecolor='none', edgecolor='green'))

        # Save the image
        plt.axis('off')
        plt.title(title)
        print("Title:", title)
        print("Save path:", save_path + title)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()

    def pad_string(self, string, length):
        while len(string) < length: string += " "
        return string
    
    def get_agent_loc(self, flatten = False):
        if flatten: return 10*self._agent_location[0] + self._agent_location[1]
        else: return self._agent_location
    
    def _get_obs(self):
        if self.full_obs: return np.array(self.get_flattened_grid()) 
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
        if "goal" in reward_dictionary.keys(): 
            self._target_location  = np.array(reward_dictionary["goal"]["location"][0])
            self._goal_identifier  = int(reward_dictionary["goal"]["identifier"])
        if "start" in reward_dictionary.keys(): 
            self._agent_location   = np.array(reward_dictionary["start"]["location"][0])
            self._agent_identifier = int(reward_dictionary['start']['identifier'])
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # print(self.masterRewardDict)
        # Reset Internal Metrics
        self.steps             = 0
        self.cumulative_reward = 0
        self.tempRewardDict    = self.masterRewardDict.copy()

        # Choose the agent's location uniformly at random if not specified
        if "start" not in self.masterRewardDict.keys(): self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else: self._agent_location = np.array(self.masterRewardDict["start"]["location"][0])

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
        reward = 0
        # We use `np.clip` to make sure we don't leave the grid bounds
        if action >= 0 and action < 5:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[int(action)]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )
            
        else: reward -= 1

        self.steps += 1
        step_reward, terminated = self.handle_events()
        reward += step_reward
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        if self.steps > self.max_steps: terminated = True
        return observation, reward, terminated, truncated, info
    
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)