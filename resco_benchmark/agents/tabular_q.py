import typing_extensions
import pickle
import os
import logging
from collections import defaultdict

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import conv2d_size_out, compute_safe_id
from resco_benchmark.agents.agent import IndependentAgent, Agent

logger = logging.getLogger(__name__)


class IQ(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            act_space = obs_act[agent_id][1]

            self.agents[agent_id] = QAgent(agent_id, act_space)


class QAgent(Agent):
    def __init__(self, agent_id, act_space):
        super().__init__()
        self.agent_id = agent_id
        self.Q = defaultdict(lambda: np.zeros(act_space))
        self._training = True
        self.num_acts = act_space
        self.last_state = None
        self.action = None

    def encode_state(self, state: np.ndarray) -> typing_extensions.Hashable:
        state = np.round(state, 0).tolist()
        return str(state)

    def act(self, observation, valid_acts=None, reverse_valid=None):
        state = self.encode_state(observation)
        # argmax = np.argmax(self.Q[state])

        try:
            argmax = np.nanargmax(np.where(self.Q[state] != 0, self.Q[state], np.nan))
        except ValueError:
            argmax = np.argmax(self.Q[state])

        self.last_state = state
        if not self._training:
            return argmax

        p = np.ones(self.num_acts) * cfg.epsilon / self.num_acts
        p[argmax] += 1.0 - cfg.epsilon
        self.action = np.random.choice(self.num_acts, p=p)
        return self.action

    def observe(self, observation, reward, done, info):
        next_state = self.encode_state(observation)

        bellman = reward + cfg.discount * max(self.Q[next_state])
        if done:
            bellman = reward

        error = bellman - self.Q[self.last_state][self.action]
        self.Q[self.last_state][self.action] += cfg.learning_rate * error
        logger.debug("TD error: {0}".format(error))

    def save(self):
        logger.debug("Saving agent {0}".format(self.agent_id))
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        # pickle.dump(self.Q, open(path + ".q", "wb"))

    def load(self):
        if cfg.load_model is None:
            raise ValueError("load_model is not set")
        agt_path = str(
            os.path.join(cfg.load_model, "agt_" + compute_safe_id(self.agent_id))
        )
        logger.debug("Loading agent {0} from {1}".format(self.agent_id, agt_path))
        self.Q = pickle.load(open(agt_path + ".q", "rb"))

        if not cfg.training:
            self.testing()

    def training(self):
        self._training = True

    def testing(self):
        logger.debug("Disabling training")
        # self.agent.training = False    TODO this breaks DQN for some reason
        self._training = False
