from typing import Any, Sequence
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

# from resco_benchmark.agents.fixed import EpsilonFixed, FixedAgent
from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.utils import conv2d_size_out, compute_safe_id
from resco_benchmark.utils.explorers import (
    LinearDecayDelayedEpsilonGreedy,
    SharedEpsGreedy,
)
from resco_benchmark.agents.agent import IndependentAgent, Agent

logger = logging.getLogger(__name__)


class IDQN(IndependentAgent):
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
            model.append(DiscreteActionValueHead())

            self.agents[agent_id] = DQNAgent(agent_id, act_space, model)


class DQNAgent(Agent):
    def __init__(self, agent_id, act_space, model, num_agents=0):
        super().__init__()
        self.agent_id = agent_id
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self._training = True

        if cfg.buffer_type == "uniform":
            self.replay_buffer = replay_buffers.ReplayBuffer(cfg.buffer_size)
        elif cfg.buffer_type == "prioritized":
            self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(cfg.buffer_size)
        else:
            raise NotImplementedError()

        # self.step_qvals = None

        if num_agents > 0:
            logger.warning("Using shared epsilon-greedy")
            explorer = SharedEpsGreedy(
                cfg.epsilon_begin,
                cfg.epsilon_end,
                num_agents * cfg.epsilon_decay_period,
                lambda: np.random.randint(act_space),
            )
        else:
            explorer = LinearDecayEpsilonGreedy(
                cfg.epsilon_begin,
                cfg.epsilon_end,
                cfg.epsilon_decay_period,
                lambda: np.random.randint(act_space),
            )

        # use_fixed_explorer = 'explorer' in cfg and cfg.explorer == 'fixed_explorer'
        # if use_fixed_explorer:
        #     self.fixed_agent = FixedAgent((None, act_space), agent_id)
        #     explorer = EpsilonFixed(
        #         cfg.epsilon_begin,
        #         cfg.epsilon_end,
        #         cfg.epsilon_decay_period,
        #         lambda: np.random.randint(act_space),
        #         self.fixed_agent
        #     )

        self.explorer = explorer

        if num_agents > 0:
            logger.warning(f"Using shared network DQN {num_agents}")
            self.agent = SharedDQN(
                self.model,
                self.optimizer,
                self.replay_buffer,
                cfg.discount,
                explorer,
                gpu=self.device.index,
                minibatch_size=cfg.batch_size,
                replay_start_size=cfg.batch_size,
                phi=lambda x: np.asarray(x, dtype=np.float32),
                target_update_interval=cfg.target_update_steps * num_agents,
                update_interval=num_agents,
            )
        else:
            self.agent = DQN(
                self.model,
                self.optimizer,
                self.replay_buffer,
                cfg.discount,
                explorer,
                gpu=self.device.index,
                minibatch_size=cfg.batch_size,
                replay_start_size=cfg.batch_size,
                phi=lambda x: np.asarray(x, dtype=np.float32),
                target_update_interval=cfg.target_update_steps,
            )

    def act(self, observation, valid_acts=None, reverse_valid=None):
        if isinstance(self.agent, SharedDQN):
            act = self.agent.act(
                observation, valid_acts=valid_acts, reverse_valid=reverse_valid
            )
        elif isinstance(self.agent, VisibleQDQN):
            act, self.step_qvals = self.agent.act(observation)
        else:
            act = self.agent.act(observation)

        return act

    def observe(self, observation, reward, done, info):
        if not self._training:
            return  # By not sending observe to agent update() is never called
        elif isinstance(self.agent, SharedDQN):
            self.agent.observe(observation, reward, done, info)
        else:
            self.agent.observe(observation, reward, done, False)

    def save(self):
        logger.debug("Saving agent {0}".format(self.agent_id))
        path = str(os.path.join(cfg.run_path, "agt_" + compute_safe_id(self.agent_id)))
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path + ".pt",
        )
        self.replay_buffer.save(path + ".replay")

    def load(self):
        if cfg.load_model is None:
            raise ValueError("load_model is not set")
        agt_path = str(
            os.path.join(cfg.load_model, "agt_" + compute_safe_id(self.agent_id))
        )
        logger.debug("Loading agent {0} from {1}".format(self.agent_id, agt_path))
        checkpoint = torch.load(agt_path + ".pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if cfg.load_replay:
            self.replay_buffer.load(agt_path + ".replay")
        if cfg.training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.testing()

    def training(self):
        self._training = True

    def testing(self):
        logger.debug("Disabling training")
        # self.agent.training = False    TODO this breaks DQN for some reason
        self._training = False


class SharedDQN(DQN):
    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
        discount: float,
        explorer: Explorer,
        gpu,
        minibatch_size,
        replay_start_size,
        phi,
        target_update_interval,
        update_interval,
    ):

        super().__init__(
            q_function,
            optimizer,
            replay_buffer,
            discount,
            explorer,
            gpu=gpu,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            phi=phi,
            target_update_interval=target_update_interval,
            update_interval=update_interval,
        )

    def act(self, obs: Any, valid_acts=None, reverse_valid=None) -> Any:
        return self.batch_act(obs, valid_acts=valid_acts, reverse_valid=reverse_valid)

    def observe(
        self,
        obs: Sequence[Any],
        reward: Sequence[float],
        done: Sequence[bool],
        reset: Sequence[bool],
    ) -> None:
        self.batch_observe(obs, reward, done, reset)

    def batch_act(
        self, batch_obs: Sequence[Any], valid_acts=None, reverse_valid=None
    ) -> Sequence[Any]:
        if valid_acts is None:
            return super(SharedDQN, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)

            batch_qvals = batch_av.params[0].detach().cpu().numpy()
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None:
                        max_val = batch_item_qval
                        max_idx = idx
                    elif batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if self.training:
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i : i + 1]
                greed = batch_argmax[i]
                if isinstance(self.explorer, SharedEpsGreedy):
                    act, greedy = self.explorer.select_action(
                        self.t,
                        lambda: greed,
                        action_value=av,
                        num_acts=len(valid_acts[i]),
                    )
                else:
                    act, greedy = self.explorer.select_action(
                        self.t, lambda: greed, action_value=av
                    )
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax

        valid_batch_action = []
        for i in range(len(batch_action)):
            valid_batch_action.append(valid_acts[i][batch_action[i]])
        return valid_batch_action


class VisibleQDQN(DQN):
    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
        discount: float,
        explorer: Explorer,
        gpu,
        minibatch_size,
        replay_start_size,
        phi,
        target_update_interval,
        agent=None,
    ):

        super().__init__(
            q_function,
            optimizer,
            replay_buffer,
            discount,
            explorer,
            gpu=gpu,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            phi=phi,
            target_update_interval=target_update_interval,
        )

        self.agent = agent
        self.action_count = 0

    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        # TODO batch support has been broken to save time
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = batch_av.greedy_actions.detach().cpu().numpy()

        # if 'fixed_mask' in cfg and cfg.fixed_mask is not None:
        #     fixed_mask = np.array(cfg[self.agent_id]['fixed_timings'], dtype=np.float32)
        #     fixed_mask[fixed_mask == 0] = -np.inf
        #     fixed_mask[fixed_mask != -np.inf] = 0
        #     qvals = batch_av.q_values[0].detach().cpu().numpy() + fixed_mask
        #     argmax_act = lambda: np.argmax(qvals)
        # else:
        argmax_act = lambda: batch_argmax[0]

        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t,
                    argmax_act,
                    action_value=batch_av[0:1],
                ),
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = [argmax_act()]
        return batch_action, batch_av

    def act(self, obs: Any) -> Any:
        self.action_count += 1
        batched, qvals = self.batch_act([obs])
        action = batched[0]
        qvals = qvals.q_values[0].detach().cpu().numpy()

        # if self.action_count % 100 == 0 and isinstance(self.explorer, EpsilonFixed):
        #     logger.debug(self.agent_id + " action count: " + str(self.action_count) + " agreement: " + str(
        #         self.explorer.agreement_count) + ' rl acts: ' + str(self.explorer.rl_acts))

        # if self.agent.agent_id == 'B3': print(qvals)

        return action, qvals
