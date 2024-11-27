import os
import time
import logging

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.logs import parse_logs
from resco_benchmark.multi_signal import MultiSignal
import resco_benchmark.mdp_options.states as states
import resco_benchmark.mdp_options.rewards as rewards
import resco_benchmark.config.optuna_objectives as objectives
from resco_benchmark.agents import *  # Import used for algorithm getattr

logger = logging.getLogger(__name__)


def main():
    if cfg.optuna_objective is not None:
        import optuna

        if cfg.optuna_trials is not None:
            callbacks = [
                optuna.study.MaxTrialsCallback(
                    int(cfg.optuna_trials), states=(optuna.trial.TrialState.COMPLETE,)
                )
            ]
        else:
            callbacks = None

        study = optuna.create_study(
            study_name=cfg.run_name,
            storage="mysql+pymysql://root@localhost/optuna",  # TODO not root
            load_if_exists=True,
            sampler=optuna.samplers.CmaEsSampler(restart_strategy="ipop"),
        )

        obj = getattr(objectives, cfg.optuna_objective)
        study.optimize(lambda trial: obj(trial, run_trial), callbacks=callbacks)
    else:
        start = time.time()
        run_trial()
        print("Time taken:", time.time() - start)


def run_trial():

    if cfg.route is not None:
        cfg.route = str(os.path.join(os.path.dirname(__file__), cfg.route))

    if cfg == "grid4x4" or cfg == "arterial4x4":
        if not os.path.exists(cfg.route):
            raise EnvironmentError(
                "You must decompress environment files defining traffic flow"
            )

    alg = None
    for module in globals().values():
        if module is not None:
            if hasattr(module, cfg.algorithm):
                alg = getattr(module, cfg.algorithm)
                break
    if alg is None:
        raise NotImplementedError("Algorithm not found")

    state_fn = getattr(states, cfg.state)
    reward_fn = getattr(rewards, cfg.reward)

    env = MultiSignal(state_fn, reward_fn)

    # Get agent id's, observation shapes, and action sizes from env
    agent = alg(env.obs_act)
    if cfg.load_model is not None:
        try:
            agent.load()
        except Exception as e:
            logger.error("Could not load model, are the RESCO parameters the same?", e)
            raise e

    if cfg.curriculum is None:
        adj_len = 1
    else:
        adj_len = len(cfg.curriculum)

    minimum_length = (
        cfg.episodes * adj_len
    )  # TODO test if cfg.testing is ok for curriculum

    for __ in range(minimum_length):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            act = agent.act(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            if cfg.algorithm == "IQ":   # TODO better to do this another way
                for signal_id in env.signals:
                    agent.agents[signal_id].action = act[signal_id]
            agent.observe(obs, rew, terminated, info)

            if terminated and env.cumulative_episode % cfg.save_frequency == 0:
                agent.save()

    agent.testing()
    for __ in range(cfg.testing):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            act = agent.act(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            agent.observe(obs, rew, terminated, info)

            if terminated and env.cumulative_episode % cfg.save_frequency == 0:
                agent.save()

    agent.save()
    env.close()
    return parse_logs()


if __name__ == "__main__":
    main()
