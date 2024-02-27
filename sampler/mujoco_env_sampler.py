import gym
import numpy as np
import wandb

class MuJoCoEnvSampler():
    def __init__(self, args, env, max_path_length=1000):
        self.env = env
        self.args = args

        self.path_length = 0
        self.total_path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.sum_reward = 0
        
        self.cur_s = None
        self.first_step = False

    def sample(self, agent, i, eval_t=False):
        self.total_path_length += 1
        if self.current_state is None:
            self.current_state = self.env.reset()
            self.first_step = True
            if self.args.env_name == 'Ant-v3':
                self.current_state = self.current_state[:27]
            elif self.args.env_name == 'Humanoid-v3':
                self.current_state = self.current_state[:45]
            self.cur_s = self.current_state.copy()
        else:
            self.first_step = False

        cur_state = self.current_state
        action = agent.select_action(cur_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)

        if 'y_velocity' in info:
            cost = np.sqrt(info["y_velocity"] ** 2 + info["x_velocity"] ** 2)
        else:
            cost = np.abs(info["x_velocity"])

        if self.args.env_name == 'Ant-v3':
            next_state = next_state[:27]
        elif self.args.env_name == 'Humanoid-v3':
            next_state = next_state[:45]
        self.path_length += 1

        reward = np.array([reward, cost])
        self.sum_reward += reward

        if terminal or self.path_length >= self.max_path_length: # NOTE
            self.current_state = None
            self.path_length = 0
            self.sum_reward = 0
        else:
            self.current_state = next_state
            self.cur_s = next_state
        return cur_state, action, next_state, reward, terminal, info
        

    def get_ter_action(self, agent):
        action = agent.select_action(self.cur_s, eval=False)
        return action
