from resco_benchmark.agents.agent import SharedAgent
from resco_benchmark.agents.maxwave import WaveAgent
from resco_benchmark.config.config import config as cfg


class MAXPRESSURE(SharedAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        self.valid_acts = cfg["valid_acts"]
        self.agent = MaxAgent(cfg["phase_pairs"])


class MaxAgent(WaveAgent):
    def act(self, observation, valid_acts=None, reverse_valid=None):
        repacked_obs = []
        for obs in observation:
            repacked_obs.append(obs[1:])
        return super().act(repacked_obs, valid_acts, reverse_valid)
