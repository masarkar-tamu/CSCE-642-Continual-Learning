from collections import deque
import numpy as np

from pfrl import replay_buffers

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import get_stats
from resco_benchmark.agents.pfrl_dqn import DQNAgent, VisibleQDQN


class CoInAgent(DQNAgent):
    def __init__(self, agent_id, act_space, model):
        super().__init__(agent_id, act_space, model)

        self.replay_buffer = CoInReplayBuffer(cfg.buffer_SIZE)

        self.bonus = 0
        self.prev_bonus_update_episode = 0
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
        self.store("QVals", cur_q.detach().cpu().numpy())

        return act

    def observe(self, observation, reward, done, info):
        self.ep_ret += reward
        self.ep_len += 1

        # Update bonus
        if self.is_new_coin_iteration():
            # Compute the bonus
            b_ = self.bonus_setting()
            if b_ is not None:
                bonus = b_
                self.replay_buffer.update_coin_rewards(bonus)
                self.cum_bonus += bonus
                self.prev_bonus_update_episode = info["eps"]

        self.agent.observe(observation, reward - self.cum_bonus, done, False)
        if done:
            self.store("EpRet", self.ep_ret)
            self.store("EpLen", self.ep_len)
            self.ep_ret = 0
            self.ep_len = 0

    def is_new_coin_iteration(self):
        # COIN iteration occurs if returns are stable
        # if "EpRet" in self.logger.keys():
        #     ep_ret_mean, ep_ret_std = get_stats(self.logger, "EpRet")
        if (
            "EpRet" in self.logger.keys()
            and len(self.logger["EpRet"]) >= 10
            and len(self.logger["EpLen"]) - self.prev_bonus_update_episode >= 10
        ):
            ep_ret_mean, ep_ret_std = get_stats(self.logger, "EpRet")
            if ep_ret_std / (abs(ep_ret_mean) + 0.00001) < cfg.epsilon_disp:
                self.prev_bonus_update_episode = len(self.logger["EpLen"])
                return True
        return False

    def bonus_setting(self):
        # COIN bonus setting
        if "QVals" in self.logger.keys():
            _, _, min_q, max_q = get_stats(self.logger, "QVals", True)
            b = 0.1 * (max_q - min_q) * (1 - cfg.discount) + cfg.epsilon_b
            self.logger.pop("QVals")
            return b
        return None


class CoInReplayBuffer(replay_buffers.ReplayBuffer):
    def update_coin_rewards(self, bonus):
        for i in range(len(self.memory)):
            self.memory[i][0]["reward"] -= bonus
