# Code adapted from https://github.com/mbodenham/gridworld-value-iteration/blob/master/deterministic.py
import numpy as np
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt


class ValueIteration(object):
    def __init__(self, args, length, rewardDictionary):
        self.args         = args
        self.step_reward  = args['sp']
        self.w            = length
        self.h            = length
        self.rewards      = rewardDictionary
        self.terminal     = []
        self.convertedRew = {}
        for _, infos in rewardDictionary.items():
            if infos["terminal"]: self.terminal.append((infos["location"][0][0] + 1) * (infos["location"][0][1] + 1) - 1)
            if infos["reward"] != 0: self.convertedRew[(infos["location"][0][0] + 1) * (infos["location"][0][1] + 1) - 1] = infos["reward"]

        # Initialize walls
        self.left_wall, self.right_wall, self.ceiling, self.floor = self.get_boundaries()

        self.state_space  = list(range(self.w * self.h))
        self.action_space = {'U': -self.w, 'D': self.w, 'L': -1, 'R': 1}
        self.actions      = ['U', 'D', 'L', 'R']
        self.P            = self.int_P()
        self.v            = ""
        self.policy       = ""
        self.gamma        = args['gamma']
        self.theta        = args['theta']
        self.env          = ""
        self.v_files      = []
        self.policy_paths = []
        self.path_files   = []
        self.path         = ""
        self.last_state   = [-1] * length**2
        self.act_map      = {"R" : 0, "U": 1, "L": 2, "D": 3}

    def get_boundaries(self):
        left  = [0 + i * self.w for i in range(self.w)]
        right = [(i+1) * self.w - 1 for i in range(self.w)]
        north = [ i for i in range(self.h)]
        south = [self.w * (self.h-1) + i for i in range(self.h)]

        return left, right, north, south

    def pretty_print(self, v):
        arrs = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(int(v[i + j*10]))
            arrs.append(row.copy())

        for row in arrs:
            print(row)

    def int_P(self):
        P = {}
        for state in self.state_space:
            for action in self.return_valid_acts(state):
                reward             = self.step_reward
                n_state            = state + self.action_space[action]

                for item, infos in self.rewards.items():
                    if n_state in self.convertedRew.keys(): 
                        reward += infos['reward']

                P[(state ,action)] = (n_state, reward)

        return P

    def return_valid_acts(self, state):
        valid_acts = []
        if state not in self.left_wall: valid_acts.append('L')
        if state not in self.right_wall: valid_acts.append('R')
        if state not in self.ceiling: valid_acts.append('U')
        if state not in self.floor: valid_acts.append('D')
        
        return valid_acts

    def check_terminal(self, state):
        if state in self.terminal: return True
        else: return False

    def check_move(self, n_state, oldState):
        if n_state not in self.state_space: return True
        elif oldState % self.m == 0 and n_state % self.m == self.m - 1: return True
        elif oldState % self.m == self.m - 1 and n_state % self.m == 0: return True
        else: return False

    def print_v(self, step = 0):
        v = np.reshape(self.v, (self.env.w, self.env.h))

        cmap = plt.get_cmap('Greens', 10)
        norm = plt.Normalize(v.min(), v.max())
        rgba = cmap(norm(v))

        for w in self.env.items.get('water').get('loc'):
            idx = np.unravel_index(w, v.shape)
            rgba[idx] = 0.0, 0.5, 0.8, 1.0


        for w in self.env.items.get('poison').get('loc'):
            idx = np.unravel_index(w, v.shape)
            rgba[idx] = 0.0, 0.0, 0.0, 1.0

        fig, ax = plt.subplots()
        im = ax.imshow(rgba, interpolation='nearest')
        print("v", v)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if v[i, j] != 0:
                    text = ax.text(j, i, '%.3f'%(v[i, j]), ha="center", va="center", color="b", fontsize = 6)

        plt.axis('off')
        filename = 'images/deterministic_v_' + str(step) + '.jpg'
        plt.title("Run " + str(step))
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        self.v_files.append(filename)
        plt.close()

    def print_policy(self, v, policy, grid, step = 0):
        v = np.reshape(v, (grid.n, grid.m))
        policy = np.reshape(policy, (grid.n, grid.m))

        cmap = plt.get_cmap('Greens', 10)
        norm = plt.Normalize(v.min(), v.max())
        rgba = cmap(norm(v))

        for w in grid.items.get('water').get('loc'):
            idx = np.unravel_index(w, v.shape)
            rgba[idx] = 0.0, 0.5, 0.8, 1.0

        for w in grid.items.get('poison').get('loc'):
            idx = np.unravel_index(w, v.shape)
            rgba[idx] = 0.0, 0.0, 0.0, 1.0
        
        fig, ax = plt.subplots()
        im = ax.imshow(rgba, interpolation='nearest')

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if v[i, j] != 0:
                    text = ax.text(j, i, policy[i, j], ha="center", va="center", color="red")

        plt.axis('off')
        file_name = 'images/deterministic_policy_' + str(z) + '.jpg'
        plt.title("Run " + str(step))
        plt.savefig(file_name, bbox_inches='tight', dpi=200)
        self.policy_paths.append(file_name)
        plt.close()

    def print_path(self):
        v = np.reshape(self.v, (self.env.w, self.env.h))
        policy = np.reshape(self.policy, (self.env.w, self.env.h))

        cmap = plt.get_cmap('Greens', 10)
        norm = plt.Normalize(v.min(), v.max())
        rgba = cmap(norm(v))

        for w in self.env.items.get('water').get('loc'):
            idx = np.unravel_index(w, v.shape)
            rgba[idx] = 0.0, 0.5, 0.8, 1.0
        
        try:
            for w in self.env.items.get('dead_area').get('loc'):
                idx = np.unravel_index(w, v.shape)
                rgba[idx] = 0.5, 0.5, 0.5, 1.0
        except:
            print("Merp")

        for w in self.env.items.get('poison').get('loc'):
            idx = np.unravel_index(w, v.shape)
            rgba[idx] = 0.0, 0.0, 0.0, 1.0

        fig, ax = plt.subplots()
        im = ax.imshow(rgba, interpolation='nearest')

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if v[i, j] != 0:
                    # text = ax.text(j, i, policy[i, j], ha="center", va="center", color="r", bbox=dict(facecolor='none', edgecolor='red'))
                    text = ax.text(j, i, policy[i, j], ha="center", va="center", color="r")
                if (i * 10 + j) in self.path:
                    text = ax.text(j, i, policy[i, j], ha="center", va="center", color="r", bbox=dict(facecolor='none', edgecolor='red'))


        plt.axis('off')
        file_name = 'images/deterministic_path_' + str(0) + '.jpg' 
        plt.title("Run " + str(0))
        plt.savefig(file_name, bbox_inches='tight', dpi=200)
        self.path_files.append(file_name)
        plt.close()

    def interate_values(self):
        converged  = False
        last_delta = 0
        while not converged:
            DELTA = 0
            for state in self.state_space:

                if self.check_terminal(state):
                    self.v[state] = 0
                else:
                    old_v         = self.v[state]
                    new_v         = []
                    base_reward = 0

                    for action in self.return_valid_acts(state):
                        (n_state, reward) = self.P.get((state, action))
                        val               = base_reward + reward + self.gamma * self.v[n_state]
                        new_v.append(val)

                    self.v[state] = max(new_v)
                    DELTA         = max(DELTA, np.abs(old_v - self.v[state]))
                    converged     = True if DELTA < self.theta else False
    
    def get_policy(self):
        for state in self.state_space:
            new_vs = []
            for action in self.return_valid_acts(state):
                (n_state, reward) = self.P.get((state, action))
                new_vs.append(reward + self.gamma * self.v[n_state])

            new_vs             = np.array(new_vs)
            best_action_idx    = np.where(new_vs == new_vs.max())[0]
            self.policy[state] = self.return_valid_acts(state)[best_action_idx[0]]

    def get_path(self):
        states_visited = []
        current_pos    = 0
        max_moves      = 100
        current_moves  = 0
        while current_moves < max_moves:
            current_moves += 1
            act = self.policy[current_pos]
            states_visited.append(current_pos)
            if act == "L":
                current_pos -= 1
            elif act == "R":
                current_pos += 1
            elif act == "D":
                current_pos += 10
            else:
                current_pos -= 10

            if current_pos == 99:
                states_visited.append(99)
                break

        return states_visited

    def make_gif(self, args, files, save_location):
        if len(files) == 0: raise ValueError("No files to save. Did you run value_iteration?")
        images = [imageio.imread(filename) for filename in files]
        imageio.mimsave(save_location, images, loop=1, fps=args['fps'])

    def reset(self):
        self.v            = np.zeros(np.prod((self.args['g_width'], self.args['g_height'])))
        self.policy       = np.full(np.prod((self.args['g_width'], self.args['g_height'])), 'n')
        self.v_files      = []
        self.policy_paths = []
        self.path_files   = []
        self.final_path   = []

    def value_iteration(self):
        self.interate_values()
        self.get_policy()
        return self.get_path()
    
    def forward(self, obs):
        grid           = obs[:-1]
        agent_loc      = int(obs[-1])

        # Save last state so that if there are no changes we dont need to val it all over again
        if (grid == self.last_state).all(): return self.act_map[self.policy[agent_loc]]
        else: self.last_state = grid
        
        # We know this should be a square grid so get the length by taking the square root
        grid_dim    = int(len(grid) ** .5)
        self.v      = np.zeros(np.prod((grid_dim, grid_dim)))
        self.policy = np.full(np.prod((grid_dim, grid_dim)), 'n')

        # Re-ravel the grid:
        square_grid = np.zeros((grid_dim, grid_dim))
        idx = 0
        for i, row in enumerate(square_grid):
            for j, column in enumerate(row):
                square_grid[i][j] = grid[idx]
                idx += 1

        self.env = square_grid

        self.value_iteration()
        return self.act_map[self.policy[agent_loc]]
        print("V", self.v)
        print("policy", self.policy)
        print(self.get_path())