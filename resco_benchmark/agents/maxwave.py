import numpy as np
from resco_benchmark.agents.agent import SharedAgent, Agent
from resco_benchmark.config.config import config as cfg


class MAXWAVE(SharedAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        self.valid_acts = cfg["valid_acts"]
        self.agent = WaveAgent(cfg["phase_pairs"])


class WaveAgent(Agent):
    def __init__(self, phase_pairs):
        super().__init__()
        self.phase_pairs = phase_pairs

    def act(self, observations, valid_acts=None, reverse_valid=None):
        acts = []
        for i, observation in enumerate(observations):
            if valid_acts is None:
                all_press = []
                for pair in self.phase_pairs:
                    all_press.append(observation[pair[0]] + observation[pair[1]])
                acts.append(np.argmax(all_press))
            else:
                max_press, max_index = None, None
                for idx in valid_acts[i]:
                    pair = self.phase_pairs[idx]
                    press = observation[pair[0]] + observation[pair[1]]
                    if max_press is None:
                        max_press = press
                        max_index = idx
                    if press > max_press:
                        max_press = press
                        max_index = idx
                acts.append(valid_acts[i][max_index])
        return acts

    def observe(self, observation, reward, done, info):
        pass
