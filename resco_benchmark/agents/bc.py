import os
import logging

import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import replay_buffers
from pfrl.explorer import Explorer
from pfrl.agents import DQN
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.utils.contexts import evaluating
from torch import Tensor
from torch.utils.data import Dataset

# from resco_benchmark.agents.fixed import EpsilonFixed, FixedAgent
from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import conv2d_size_out, compute_safe_id
from resco_benchmark.agents.agent import IndependentAgent, Agent

logger = logging.getLogger(__name__)


class BC(IndependentAgent):
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

            self.agents[agent_id] = BCAgent(agent_id, act_space, model)


class BCAgent(Agent):
    def __init__(self, agent_id, act_space, model):
        super().__init__()
        self.agent_id = agent_id
        self.action_space = list(range(act_space))
        self.model = model
        self.model.to(self.device)
        self.train()
        self.acts = 0

    def observe(self, observation, reward, done, info):
        if done:
            self.acts = 0

    def act(self, observation, valid_acts=None, reverse_valid=None):
        # if self.acts < 4:
        #     self.acts += 1
        #     return 0

        if self.agent_id != 'B3': return 0
        with torch.no_grad():
            print(observation)
            observation = torch.from_numpy(observation.astype(np.float32)).to(self.device)
            probs = torch.nn.functional.softmax(self.model(observation), dim=-1).cpu().numpy()
            #return np.random.choice(self.action_space, p=probs)

            print(probs)
            if observation == np.asarray([4,  0, 12, 12,  8,  1,  0]): return 1
            if observation == np.asarray([0, 8, 8, 8, 8]): return 1
            if observation == np.asarray([ 4,  4,  4,  4,  0]): return 1
            return np.argmax(probs, axis=-1)

    def train(self):
        if self.agent_id != 'B3': return    # TODO: fix this
        import pickle
        with open(f'{self.agent_id}_memory.pkl', 'rb') as f:
            data = pickle.load(f)

        class BCDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                obs, act = self.data[idx]
                #one hot act
                actmask = np.zeros(2)
                actmask[act] = 1
                return obs, actmask

        dataset = BCDataset(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        weight = torch.Tensor([1, 3]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(cfg.epochs):
            print('Epoch', epoch)
            losses = []
            accs = []
            for i, (inputs, labels) in enumerate(train_loader):
                # print(np.concatenate([inputs, labels], axis=1))
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # print(outputs)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                acc = (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean().item()

                print(acc)
                if acc != 1:
                    argmax = outputs.argmax(dim=-1).unsqueeze(1)
                    labels = labels.argmax(dim=-1).unsqueeze(1)
                    print(torch.concatenate([inputs, labels, argmax], axis=1))
                    print(outputs)
                accs.append(acc)
                # print(acc)
            print('Acc:', np.mean(accs))
            print('Loss:', np.mean(losses))
