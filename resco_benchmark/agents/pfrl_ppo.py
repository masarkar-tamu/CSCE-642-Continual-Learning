import os
import numpy as np

import torch
import torch.nn as nn

from pfrl.nn import Branched
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import lecun_init, conv2d_size_out, compute_safe_id
from resco_benchmark.agents.agent import IndependentAgent, Agent


class IPPO(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = PFRLPPOAgent(agent_id, obs_space, act_space)


class PFRLPPOAgent(Agent):
    def __init__(self, agent_id, obs_space, act_space):
        super().__init__()
        self.agent_id = agent_id

        model = nn.Sequential()
        if len(obs_space) == 1:
            input_size = obs_space[0]
        else:
            height = conv2d_size_out(obs_space[1])
            width = conv2d_size_out(obs_space[2])
            input_size = height * width * cfg.number_of_units

            model.append(
                lecun_init(nn.Conv2d(obs_space[0], cfg.number_of_units, kernel_size=(2, 2)))
            )
            if "linear_model" not in cfg:
                model.append(nn.ReLU())
            model.append(nn.Flatten())

        model.append(lecun_init(nn.Linear(input_size, cfg.number_of_units)))
        if "linear_model" not in cfg:
            model.append(nn.ReLU())
        for i in range(cfg.number_of_layers):
            model.append(lecun_init(nn.Linear(cfg.number_of_units, cfg.number_of_units)))
            if "linear_model" not in cfg:
                model.append(nn.ReLU())

        model.append(
            Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(cfg.number_of_units, act_space), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(cfg.number_of_units, 1)),
            )
        )
        self.model = model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
        )
        self.agent = PPO(
            self.model,
            self.optimizer,
            gpu=self.device.index,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            clip_eps=cfg.clip_eps,
            update_interval=cfg.update_interval,
            minibatch_size=cfg.batch_size,
            epochs=cfg.epochs,
            standardize_advantages=True,
            entropy_coef=cfg.entropy_coef,
            max_grad_norm=cfg.max_grad_norm,
        )

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self):
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path + ".pt",
        )

    def load(self):
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        self.model.load_state_dict(
            torch.load(path + ".pt", map_location=self.device)["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(path + ".pt", map_location=self.device)["optimizer_state_dict"]
        )
        if not cfg.training:
            self.agent.training = False
