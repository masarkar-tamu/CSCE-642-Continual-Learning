import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.fixed import FIXED


# Config values of phase or null will use direct 'set next phase' actions


#   Replaces action space with simplified set of [keep, increase, decrease] following fixed timing config
class FixedCyclePlan:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.num_acts = 3

    def act(self, acts):
        for signal in self.signals:
            agt_act = acts[signal]
            if agt_act == 0:
                pass
            elif agt_act == 1:
                self.fixed_agent.agents[signal].increase_current_phase_length()
            elif agt_act == 2:
                self.fixed_agent.agents[signal].decrease_current_phase_length()
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
        return acts


# Follow config fixed time defined cycle, choose to stay in current phase or go next
class FixedCycle:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.num_acts = 2

        # Force all phases to 1 step length
        for signal in self.signals:
            for i in range(len(self.fixed_agent.agents[signal].plan)):
                self.fixed_agent.agents[signal].plan[i] = 1

    def act(self, acts):
        for signal in self.signals:
            agt_act = acts[signal]
            if agt_act == 0:  # Keep same phase
                self.fixed_agent.agents[signal].active_phase_len = 0
            elif agt_act == 1:  # Go next
                self.fixed_agent.agents[signal].active_phase_len = np.inf
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
        return acts


# Continuous or discrete, choose current phase's length
class PhaseLength:
    pass  # TODO
