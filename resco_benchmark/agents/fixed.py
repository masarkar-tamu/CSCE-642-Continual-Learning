import os
import random

import numpy as np
from pfrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from pfrl.explorers.epsilon_greedy import select_action_epsilon_greedily

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.utils.utils import permutations_without_rotations, compute_safe_id
import logging

logger = logging.getLogger(__name__)


class FIXED(IndependentAgent):
    def __init__(self, obs_act, dumb=False):
        super().__init__(obs_act)
        for key in obs_act:
            self.agents[key] = FixedAgent(obs_act[key], key, dumb=dumb)


class FixedAgent(Agent):
    def __init__(self, obs_act, key, dumb=False):
        super().__init__()
        self.agent_id = key
        self.acts = 0
        self.curr_idx = 0
        self.plan = list()
        self.offset = 0
        self.active_phase = 0
        self.active_phase_len = 0
        num_acts = obs_act[1]

        if not dumb:  # Use configuration settings
            timings = cfg[key]["fixed_timings"]
            for i in range(num_acts):
                self.plan.append(int(timings[i]))

            phase_orders = np.asarray(
                list(permutations_without_rotations(range(num_acts)))
            )
            self.phase_order = phase_orders[cfg[key]["fixed_phase_order_idx"]].tolist()

            if "fixed_offset" in cfg[key]:
                self.offset = cfg[key]["fixed_offset"]
                if self.agent_id != "B3":
                    logger.warning(
                        "Fixed agent offset not implemented for " + self.agent_id
                    )
        else:
            self.plan = [4] * num_acts
            self.phase_order = range(num_acts)
            self.offset = 0
        self.cycle_count = 0
        self.interference_indices = list()
        self.num_acts = num_acts

    def act(self, _=None):
        if np.all(np.array(self.plan) == 0):
            self.plan[self.active_phase] = 1  # Ensure that at least one phase is active
        if self.agent_id == "B3" and self.acts == 0:  # TODO rewrite?
            tmp_offset = self.offset
            while tmp_offset > 0:
                if self.active_phase_len > tmp_offset:
                    self.active_phase_len -= tmp_offset
                    tmp_offset = 0
                else:
                    self.active_phase = (self.active_phase - 1) % len(self.plan)
                    swap = self.plan[self.active_phase]
                    swap = swap - tmp_offset
                    if swap < 0:
                        tmp_offset = swap * -1
                    else:
                        self.active_phase_len = swap
                        tmp_offset = 0

        if self.active_phase_len >= np.abs(self.plan[self.active_phase]):
            self.active_phase = (self.active_phase + 1) % len(self.plan)
            while self.plan[self.active_phase] == 0:  # Skip phases with 0 duration
                self.active_phase = (self.active_phase + 1) % len(self.plan)
            self.active_phase_len = 1
        else:
            self.active_phase_len += 1

        self.acts += 1

        if cfg.step_off != 0 and (self.acts % len(self.plan) == 0 or self.acts == 1):
            self.cycle_count = 0
            self.interference_indices = sorted(random.sample(list(range(sum(self.plan))), cfg.step_off))

        if len(self.interference_indices) != 0 and self.cycle_count == self.interference_indices[0]:
            self.interference_indices.pop(0)
            self.cycle_count += 1
            eps_acts = list(range(self.num_acts))
            eps_acts.remove(self.active_phase)
            eps_act = random.choice(eps_acts)
            if cfg.cycle_forward:
                self.active_phase_len = 0
                self.active_phase = eps_act
            return eps_act
        self.cycle_count += 1

        return self.phase_order[self.active_phase]

    def __getitem__(self, act):
        return self.plan[self.phase_order[act]]

    def increase_current_phase_length(self, size=1):
        self.plan[self.active_phase] += size

    def decrease_current_phase_length(self, size=1):
        self.plan[self.active_phase] -= size
        if self.plan[self.active_phase] < 0:
            self.plan[self.active_phase] = 0

    def save(self):
        with open(
            os.path.join(
                cfg.run_path, "fixd_{0}.txt".format(compute_safe_id(self.agent_id))
            ),
            "w",
        ) as f:
            f.write(str(self.plan) + "\n")
            f.write(str(self.phase_order) + "\n")
            f.write(str(self.offset) + "\n")

    def observe(self, observation, reward, done, info):
        pass


class EpsilonFixed(LinearDecayEpsilonGreedy):
    def __init__(
        self, start_epsilon, end_epsilon, decay_steps, random_action_func, fixed_agent
    ):
        super().__init__(start_epsilon, end_epsilon, decay_steps, random_action_func)
        self.fixed_agent = fixed_agent
        self.rl_acts = 0
        self.agreement_count = 0

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)

        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )

        b, reroll = select_action_epsilon_greedily(
            0.5, self.random_action_func, greedy_action_func
        )

        fixed_action = self.fixed_agent.act()
        if greedy:
            if reroll:
                a = b
            else:
                self.rl_acts += 1
                if fixed_action != a:
                    self.fixed_agent.active_phase = a
                    self.fixed_agent.active_phase_len = 1
                else:
                    self.agreement_count += 1
        else:
            a = fixed_action

        return a
