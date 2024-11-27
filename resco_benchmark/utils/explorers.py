import logging
import numpy as np

from pfrl import explorers
from pfrl.explorer import Explorer
from pfrl.explorers.epsilon_greedy import select_action_epsilon_greedily

logger = logging.getLogger(__name__)


class SharedEpsGreedy(explorers.LinearDecayEpsilonGreedy):

    def select_action(self, t, greedy_action_func, action_value=None, num_acts=None):
        self.epsilon = self.compute_epsilon(t)
        if num_acts is None:
            fn = self.random_action_func
        else:
            fn = lambda: np.random.randint(num_acts)
        a, greedy = select_action_epsilon_greedily(self.epsilon, fn, greedy_action_func)
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        if num_acts is None:
            return a
        else:
            return a, greedy


class LinearDecayDelayedEpsilonGreedy(Explorer):
    def __init__(
        self, start_epsilon, end_epsilon, decay_steps, random_action_func, delay
    ):
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.epsilon = start_epsilon
        self.delay = delay
        self.greedy_traj = False
        self.greedy_cnt = 0
        self.current_act = None

        self.off_policy_acts = 0

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        if self.decay_steps == 0:
            self.epsilon = self.end_epsilon
        else:
            self.epsilon = self.compute_epsilon(t)

        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )

        if self.delay == 1:  # Do standard eps-greedy
            if not greedy:
                self.off_policy_acts += 1
            self.current_act = a
            return a

        random_perc = 0.5
        k = (
            self.epsilon / (self.delay - self.epsilon * self.delay - 0.0000000000001)
        ) * (1 / random_perc)
        a, greedy = select_action_epsilon_greedily(
            k, self.random_action_func, greedy_action_func
        )

        if self.greedy_traj and self.greedy_cnt == self.delay:
            self.greedy_traj = False
            self.greedy_cnt = 0
        elif self.greedy_traj:
            self.greedy_cnt += 1
            if self.current_act is not None:
                self.off_policy_acts += 1
                return self.current_act

        if not greedy and np.random.rand() < random_perc:
            self.greedy_traj = True
            self.greedy_cnt = 1
            if self.current_act is not None:
                self.off_policy_acts += 1
                return self.current_act
        if not greedy:
            self.off_policy_acts += 1
        self.current_act = a
        return a

    def __repr__(self):
        return "LinearDecayDelayedEpsilonGreedy(epsilon={})".format(self.epsilon)
