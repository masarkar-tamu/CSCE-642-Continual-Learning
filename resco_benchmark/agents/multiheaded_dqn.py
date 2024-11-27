from collections import deque, defaultdict

import torch.nn as nn

from pfrl.q_functions import DiscreteActionValueHead

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import conv2d_size_out
from resco_benchmark.agents.agent import IndependentAgent
from resco_benchmark.agents.pfrl_dqn import DQNAgent


def queue_percentage(signals):
    signal_queues = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        queue_percents = list()
        for lane_id in signal.observation.lanes:
            lane_max = 0
            for vehicle in signal.observation.lanes[lane_id].vehicles.values():
                pos = vehicle.position
                if pos > lane_max and vehicle.queued:
                    lane_max = pos
            if signal.lane_lengths[lane_id] < cfg.max_distance:
                queue_percents.append(lane_max / signal.lane_lengths[lane_id])
            else:
                queue_percents.append(lane_max / cfg.max_distance)
        signal_queues[signal_id] = queue_percents
    return signal_queues


class IMultiDQN(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        self.alt_agents = dict()
        self.wait_deque: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=int(12))
        )
        self.diff_deque: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=int(12))
        )
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            model = nn.Sequential()
            if len(obs_space) == 1:
                model.append(nn.Linear(obs_space[0], cfg.number_of_units))
            else:
                h = conv2d_size_out(obs_space[1])
                w = conv2d_size_out(obs_space[2])

                model.append(nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)))
                model.append(nn.ReLU())
                model.append(nn.Flatten())
                model.append(
                    nn.Linear(h * w * cfg.number_of_units, cfg.number_of_units)
                )

            model.append(nn.ReLU())
            for i in range(cfg.number_of_layers):
                model.append(nn.Linear(cfg.number_of_units, cfg.number_of_units))
                model.append(nn.ReLU())
            model.append(nn.Linear(cfg.number_of_units, act_space))
            model.append(DiscreteActionValueHead())

            self.agents[key] = DQNAgent(key, act_space, model)
            self.alt_agents[key] = DQNAgent(key + "_alt", act_space, model)

        self.switch_heads_trigger = dict()
        self.num_alt = 0
        self.num_agt = 0
        self.switch_timer = 0

    def act(self, observation):
        acts = dict()
        alt_acts = dict()
        for agent_id in observation.keys():
            acts[agent_id] = self.agents[agent_id].act(observation[agent_id])
            alt_acts[agent_id] = self.alt_agents[agent_id].act(observation[agent_id])

        if len(self.switch_heads_trigger) != 0:
            for agent_id in observation.keys():
                if self.switch_heads_trigger[agent_id]:
                    self.num_alt += 1
                    acts[agent_id] = alt_acts[agent_id]
                    self.agents[agent_id].agent.batch_last_action = [alt_acts[agent_id]]
                else:
                    self.num_agt += 1
                    self.alt_agents[agent_id].agent.batch_last_action = [acts[agent_id]]

        return acts

    def observe(self, observation, reward, done, info):
        for agent_id in observation.keys():
            self.alt_agents[agent_id].observe(
                observation[agent_id], info["alt_reward"][agent_id], done, info
            )

        waits = info["wait"]
        for agent_id in observation.keys():
            self.wait_deque[agent_id].append(waits[agent_id])
            # Compute average rate of change in the queue
            qp = queue_percentage(info["signals"])
            max_q = max(qp[agent_id])
            if max_q > 0.8:
                self.switch_heads_trigger[agent_id] = True
                self.switch_timer = 0
            else:
                self.switch_timer += 1
                self.switch_heads_trigger[agent_id] = False

        for agent_id in observation.keys():
            self.agents[agent_id].observe(
                observation[agent_id], reward[agent_id], done, info
            )
