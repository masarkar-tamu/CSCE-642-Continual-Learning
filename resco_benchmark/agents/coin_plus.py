from collections import deque
import numpy as np
import torch
import torch.nn as nn
from pfrl import replay_buffers
from pfrl.q_functions import DiscreteActionValueHead

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import get_stats, conv2d_size_out
from resco_benchmark.agents.pfrl_dqn import DQNAgent, VisibleQDQN
from resco_benchmark.agents.agent import IndependentAgent
from resco_benchmark.agents.pfrl_dqn import IDQN

class ICoInPlus(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]

            model = nn.Sequential()
            if len(obs_space) == 1:
                input_size = obs_space[0]
            else:
                height = conv2d_size_out(obs_space[1])
                width = conv2d_size_out(obs_space[2])
                input_size = height * width * cfg.number_of_units

                model.append(
                    nn.Conv2d(obs_space[0], cfg.number_of_units, kernel_size=(2, 2))
                )
                if "linear_model" not in cfg:
                    model.append(nn.ReLU())
                model.append(nn.Flatten())

            model.append(nn.Linear(input_size, cfg.number_of_units))
            if "linear_model" not in cfg:
                model.append(nn.ReLU())
            for i in range(cfg.number_of_layers):
                model.append(nn.Linear(cfg.number_of_units, cfg.number_of_units))
                if "linear_model" not in cfg:
                    model.append(nn.ReLU())
            model.append(nn.Linear(cfg.number_of_units, act_space))
            model.append(DiscreteActionValueHead())

            self.agents[agent_id] = CoInPAgent(agent_id, act_space, model)

class CoInPAgent(DQNAgent):
    def __init__(self, agent_id, act_space, model):
        super().__init__(agent_id, act_space, model)

        self.replay_buffer = CoInReplayBuffer(cfg.buffer_size)

        self.bonus = 0.005 # hardcoded for now based on coin plus code
        self.decay = 0.999 # Hardcoded " "
        #self.prev_bonus_update_episode = 0
        self.cum_bonus = self.bonus

        self.logger = dict()
        self.ep_ret = 0
        self.ep_len = 0

        self.agent = VisibleQDQN(
            self.model,
            self.optimizer,
            self.replay_buffer,
            cfg.discount,
            self.explorer,
            gpu=self.device.index,
            minibatch_size=cfg.batch_size,
            replay_start_size=cfg.batch_size,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            target_update_interval=cfg.target_update_steps,
        )

    def store(self, key, value):
        if key not in self.logger:
            if "EpRet" == key:
                self.logger[key] = deque(maxlen=10)
                self.logger[key].append(value)
            else:
                self.logger[key] = [value]
        else:
            self.logger[key].append(value)

    def act(self, observation, valid_acts=None, reverse_valid=None):
        act, cur_q = self.agent.act(observation)

        # Current Q-value estimates
        self.store("QVals", cur_q)

        return act
    
    def observe(self, observation, reward, done, info):
        self.ep_ret += reward
        self.ep_len += 1

        self.bonus *= self.decay
        self.cum_bonus += self.bonus

        self.agent.observe(observation, reward - self.cum_bonus, done, False)
        if done: 
            self.store("EpRet", self.ep_ret)
            self.store("EpLen", self.ep_len)
            self.ep_ret = 0
            self.ep_len = 0
            

class CoInReplayBuffer(replay_buffers.ReplayBuffer):
    def update_coin_rewards(self, bonus):
        for i in range(len(self.memory)):
            self.memory[i][0]["reward"] -= bonus
