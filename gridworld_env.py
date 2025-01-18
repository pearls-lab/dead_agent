# Based on https://gymnasium.farama.org/introduction/create_custom_env/
from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import utils
from distutils.util import strtobool
import pickle


class GridWorldEnv(gym.Env):

    def __init__(self, args, size: int = 5):
        self.args                 = args
        # The size of the square grid. Prob need to switch this to just pull from the arguments
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
            0: np.array([0, 1]),  # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),  # down
        }

        # Internal metrics
        self.steps                = 0
        self.cumulative_reward    = 0
        self.masterRewardDict     = {}
        self.tempRewardDict       = {}
        self.full_obs             = True
        self.event_locations      = []
        self.min_dist             = 5
        self.max_steps            = args['max_env_steps']
        self.locIdenDict          = []
        self.eval_env             = False
        self.allIdentifiers       = [self._agent_identifier, self._goal_identifier]
        for _, value in self.masterRewardDict.items():
            info       = value
            identifier = np.random.randint(1, 100)
            while identifier in self.allIdentifiers: identifier = np.random.randint(1, 100)
            self.allIdentifiers.append(identifier)
            for location in info['location']: self.locIdenDict[location] = identifier
        
        # If > 0: then agent will die when timer hits 0
        self.active_events         = {}
        self.all_locs              = []
        self.game_locs             = []
        self.last_ten_game_locs    = []
        self.all_traj              = []
        self.current_traj          = []
        self.current_acts          = []
        self.subob_traj            = []
        self.total_resets          = 0
        self.use_teacher_forcing   = args['teacher_force']
        self.use_walkthrough       = False
        

    def print_target_agent(self):
        print("Agent:", self._agent_location)
        print("Target:", self._target_location)

    def plot_game_loc_diversity(self, initial_path, algo,  only_text):
        utils.plot_array_and_save(
                    self.last_ten_game_locs, 
                    initial_path + "_buffer_diversity", title = algo + " buffer div", 
                    x_label = "episodes", y_label = "uniq locations", y_max = 100, only_text = only_text)

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

        rgba[location[0]][location[1]] = 0.0, 0.5, 0.8, 1.0
        colored_spots.append(location)


        if 'goal' in self.masterRewardDict.keys(): location = self.masterRewardDict['goal']['location'][0]
        else:                                      location = self._target_location

        rgba[location[0]][location[1]] = 0.8, 0.5, 0.0, 1.0
        colored_spots.append(location)

        if 'dead_areas' in self.masterRewardDict.keys():
            for location in self.masterRewardDict['dead_areas']['location']:
                rgba[location[0]][location[1]] = 0.5, 0.5, 0.5, 1.0
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
    
    def load_rewards(self, reward_dictionary, eval_env = False):
        # Initializes master and temp reward dictionary
        # Temp is 'reloaded' from master when the env resets
        self.eval_env         = eval_env
        self.masterRewardDict = reward_dictionary.copy()
        self.tempRewardDict   = self.masterRewardDict.copy()
        self.event_locations  = []
        for _, info in reward_dictionary.items():
            for location in info['location']:
                self.event_locations.append(self.get_flat_loc(location))
        if "goal" in reward_dictionary.keys(): 
            self._target_location  = np.array(reward_dictionary["goal"]["location"][0])
            self._goal_identifier  = int(reward_dictionary["goal"]["identifier"])
        if "start" in reward_dictionary.keys(): 
            self._agent_location   = np.array(reward_dictionary["start"]["location"][0])
            self._agent_identifier = int(reward_dictionary['start']['identifier'])

    def get_flat_loc(self, location):    
        return int(location[0] * 10 + location[1])
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # print(self.masterRewardDict)
        # Reset Internal Metrics

        # Calling it teacher-forcing but prob need a better name for this
        if self.use_teacher_forcing:
            # If the agent is not currently using the walkthrough, passively check for if a better trajectory is generated
            if not self.use_walkthrough:
                if self.cumulative_reward > 0:
                    # If the suboptimal trajectory has not been initialized, just take the first one that returns a reward. Otherwise, use length as heuristic
                    if len(self.subob_traj) < 1 or (len(self.subob_traj) > len(self.current_acts)):
                        self.subob_traj = self.current_acts.copy()

                # Every 100 non-walkthrough games, use the most optimal trajectory found once
                non_walkthrough_games = 100        
                self.total_resets += 1
                if self.total_resets % non_walkthrough_games == 0:
                    self.use_walkthrough = True
            else:
                self.use_walkthrough = False

        self.current_acts      = []
        self.steps             = 0
        self.cumulative_reward = 0
        self.tempRewardDict    = self.masterRewardDict.copy()
        self.active_events     = {}

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
        if len(self.game_locs) >= 100:
            self.last_ten_game_locs.append(len(list(utils.flatten_and_count_unique(self.game_locs).keys())))
            self.game_locs.pop(0)
        self.game_locs.append(self.all_locs)
        self.all_locs = [0]
        # print("Total locations: ", len(self.all_locs))
        self.all_traj.append(self.current_traj.copy())
        self.current_traj = [observation]
        return observation, info
    
    def handle_events(self):
        # TODO: Do handling for special events or something
        reward_at_current_step = -0.01
        terminal               = False
        flat_agent_loc         = self.get_flat_loc(self._agent_location)

        if 'dying' in self.active_events:
            if self.active_events['dying'] <= 0:
                return -10, True
            else:
                self.active_events['dying'] -= 1

        if flat_agent_loc not in self.event_locations: return reward_at_current_step, terminal
        
        for key, info in self.masterRewardDict.items():
            for location in info["location"]:
                if (self._agent_location == location).all(): 
                    if key == 'dead_areas': 
                        if 'dying' not in self.active_events.keys():
                            if self.args['death_timer'] > 0: self.active_events['dying'] = self.args['death_timer']
                            else:                            self.active_events['dying'] = np.random.randint(1, (self.args['death_timer'] * -1) + 1)
                    else:
                        reward_at_current_step   += info['reward']
                        info['available']         = False
                    if not terminal: 
                        terminal = bool(strtobool(info['terminal'])) # terminal event + non terminal event = terminal

        return reward_at_current_step, terminal

    def step(self, action):
        if self.use_walkthrough and len(self.subob_traj) > 0: # This should always be false with no teacher forcing
            action = self.subob_traj[self.steps]
        reward = 0
        # We use `np.clip` to make sure we don't leave the grid bounds
        if action >= 0 and action < 5:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[int(action)]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )
            
        else: reward -= 1  
        self.current_acts.append(action)
        self.steps             += 1
        step_reward, terminated = self.handle_events()
        reward                 += step_reward
        truncated               = False
        observation             = self._get_obs()
        info                    = self._get_info()
        info['died']            = False
        if reward < -1:
            info['died']        = True
        self.all_locs.append(self.get_flat_loc(self._agent_location))
        if self.steps > self.max_steps: terminated = True
        self.cumulative_reward += reward
        self.current_traj.append(observation)
        assert len(self.current_traj) == self.steps + 1
        return observation, reward, terminated, truncated, info
    
    def save_trajectories(self):
        with open('traj.pkl', 'wb') as f:
            pickle.dump(self.all_traj, f)

    
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)