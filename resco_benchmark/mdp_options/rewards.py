from __future__ import annotations
import logging

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.traffic_signal import Signal


logger = logging.getLogger(__name__)


def oracle_delay(signals: dict[str, Signal]) -> dict[str, float]:
    # Will include timeLoss of vehicles not even on a road with an intersection, etc. + departure delay
    rewards: dict[str, float] = dict()
    total_reward: float = 0.0
    a_signal = list(signals.values())[0]

    for veh_id in a_signal.sumo.vehicle.getIDList():
        total_reward -= a_signal.sumo.vehicle.getTimeLoss(veh_id)

    for signal_id in signals:
        rewards[signal_id] = total_reward
    return rewards


def oracle_delay_depart(signals: dict[str, Signal]) -> dict[str, float]:
    # Will include timeLoss of vehicles not even on a road with an intersection, etc. + departure delay
    rewards: dict[str, float] = dict()
    total_reward: float = 0.0
    a_signal = list(signals.values())[0]

    for veh_id in a_signal.sumo.vehicle.getIDList():
        if a_signal.sumo.vehicle.getRouteIndex(veh_id) == -1:  # Not departed yet
            total_reward -= a_signal.sumo.vehicle.getDepartDelay(veh_id)
        else:
            total_reward -= a_signal.sumo.vehicle.getTimeLoss(veh_id)

    for signal_id in signals:
        rewards[signal_id] = total_reward
    return rewards


def fixed_tac(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]

        location = list()
        for act in range(len(signal.green_phases)):
            if signal.fixed_agent[act] != 0:
                location.append(signal.time_since_phase[act])
        # cycle_length = sum(signal.fixed_agent.plan) * cfg.step_length
        # location.append(signal.current_phase)
        location = np.asarray(location)

        goals = np.linalg.norm(location - signal.fixed_timers, axis=1)
        cl = np.argmin(goals)
        fr = np.argmax(goals)
        closest_goal = signal.fixed_timers[cl]
        furthest_goal = signal.fixed_timers[fr]
        close_dist = goals[cl]
        far_dist = goals[fr]
        if close_dist == 0 or far_dist == 0:
            rewards[signal_id] = 0
            continue

        v1_u = closest_goal / np.linalg.norm(closest_goal)
        v2_u = furthest_goal / np.linalg.norm(furthest_goal)
        close_far_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        fixed_length = np.sqrt(
            far_dist ** 2
            + close_dist ** 2
            - (2 * far_dist * close_dist * np.cos(close_far_angle))
        )
        if fixed_length < 1e-3:
            rewards[signal_id] = 0
            continue

        close_fixed_angle = (close_dist ** 2 + fixed_length ** 2 - far_dist ** 2) / (
                2 * close_dist * fixed_length
        )
        if close_fixed_angle >= 1 or close_fixed_angle <= -1:
            close_fixed_angle = 0
        else:
            close_fixed_angle = np.arccos(close_fixed_angle)

        far_fixed_angle = (far_dist ** 2 + fixed_length ** 2 - close_dist ** 2) / (
                2 * far_dist * fixed_length
        )
        if far_fixed_angle >= 1 or far_fixed_angle <= -1:
            far_fixed_angle = 0
        else:
            far_fixed_angle = np.arccos(far_fixed_angle)

        reward = far_fixed_angle
        reward /= close_far_angle if close_far_angle != 0 else 1
        reward /= close_fixed_angle if close_fixed_angle != 0 else 1
        reward *= -1
        rewards[signal_id] = reward
    return rewards


def throughput(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        rewards[signal_id] = signals[signal_id].observation.departed
    return rewards


def speed(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        cum_speed: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            for vehicle in lane.vehicles.values():
                cum_speed += vehicle.average_speed
        rewards[signal_id] = cum_speed
    return rewards


def acceleration(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        accel: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            for vehicle in lane.vehicles.values():
                accel += abs(vehicle.average_acceleration) + 1
        rewards[signal_id] = (
            -accel * signals[signal_id].observation.total_wait
        ) / 10000.0
    return rewards


def fuel(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        cum_fuel: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            for vehicle in lane.vehicles.values():
                cum_fuel += vehicle.fuel_consumption
        rewards[signal_id] = -cum_fuel
    return rewards


def delay(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        sig_reward: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            for vehicle in lane.vehicles.values():
                sig_reward += vehicle.delay
        rewards[signal_id] = sig_reward
    return rewards


def maxqueue(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_queue: int = 0
        for lane in signals[signal_id].observation.lanes.values():
            if lane.queued > max_queue:
                max_queue = lane.queued

        rewards[signal_id] = -max_queue
    return rewards


def maxqueue_maxwait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_queue: int = 0
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            if lane.queued > max_queue:
                max_queue = lane.queued

            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait

        rewards[signal_id] = -(max_wait * max_queue)
    return rewards


def avgqueue_maxwait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        sum_queue: int = 0
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            sum_queue += lane.queued

            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait
        sum_queue /= len(signals[signal_id].observation.lanes)
        rewards[signal_id] = -(max_wait * sum_queue)
    return rewards


def avgqueue_maxdelay(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        sum_queue: int = 0
        max_delay: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            sum_queue += lane.queued

            for veh in lane.vehicles.values():
                delay = -veh.delay
                if delay > max_delay:
                    max_delay = delay
        sum_queue /= len(signals[signal_id].observation.lanes)
        rewards[signal_id] = -(max_delay * sum_queue)
    return rewards


def maxqueue_maxwait_div(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_queue: int = 0
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            if lane.queued > max_queue:
                max_queue = lane.queued

            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait
        if max_queue != 0:
            rewards[signal_id] = -(max_wait / max_queue)
        else:
            rewards[signal_id] = 0
    return rewards


def maxqueue_maxdelay_div(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_queue: int = 0
        max_delay: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            if lane.queued > max_queue:
                max_queue = lane.queued

            for veh in lane.vehicles.values():
                delay = -veh.delay
                if delay > max_delay:
                    max_delay = delay
        if max_queue != 0:
            rewards[signal_id] = -(max_delay / max_queue)
        else:
            rewards[signal_id] = 0
    return rewards


def maxwaitqueue_wait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_wait: float = 0.0
        max_queue: int = 0
        for lane in signals[signal_id].observation.lanes.values():
            if lane.queued > max_queue:
                max_queue = lane.queued

            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait

        reward = max_wait * max_queue * signals[signal_id].observation.total_wait
        rewards[signal_id] = -reward
    return rewards


def queue(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        rewards[signal_id] = -signals[signal_id].observation.total_queued
    return rewards


def wait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        rewards[signal_id] = -signals[signal_id].observation.total_wait
    return rewards


def maxwait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():

            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait

        rewards[signal_id] = -max_wait
    return rewards


def maxwaitadj(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            ordered_vehicles = dict()
            for veh in lane.vehicles.values():
                ordered_vehicles[veh.position] = veh.wait
            sorted_vehicles = sorted(ordered_vehicles.keys())

            for i, veh in enumerate(sorted_vehicles):
                wait = ordered_vehicles[veh]
                adj_value = wait * (len(sorted_vehicles) - (i + 1))
                if adj_value > max_wait:
                    max_wait = adj_value

        rewards[signal_id] = -max_wait
    return rewards


def maxdelay(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_delay: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():

            for veh in lane.vehicles.values():
                delay = -veh.delay
                if delay > max_delay:
                    max_delay = delay

        rewards[signal_id] = -max_delay
    return rewards


def maxdelayadj(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_delay: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            ordered_vehicles = dict()
            for veh in lane.vehicles.values():
                ordered_vehicles[veh.position] = -veh.delay
            sorted_vehicles = sorted(ordered_vehicles.keys())

            for i, veh in enumerate(sorted_vehicles):
                delay = ordered_vehicles[veh]
                adj_value = delay * (len(sorted_vehicles) - (i + 1))
                if adj_value > max_delay:
                    max_delay = adj_value

        rewards[signal_id] = -max_delay
    return rewards


def maxwait_logwait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait

        reward: float = 0.0
        if signals[signal_id].observation.total_wait != 0.0:
            reward = max_wait * np.log(
                signals[signal_id].observation.total_wait
            )  # / signals[signal_id].observation.total_wait
        rewards[signal_id] = -reward
    return rewards


def maxwait_wait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        max_wait: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            for veh in lane.vehicles.values():
                if veh.wait > max_wait:
                    max_wait = veh.wait

        reward: float = 0.0
        if signals[signal_id].observation.total_wait != 0.0:
            reward = max_wait * signals[signal_id].observation.total_wait
        rewards[signal_id] = -reward
    return rewards


def pressure(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        entering_queued: int = signal.observation.total_queued

        exiting_queued: int = 0
        for lane_id in signal.outbound_lanes:
            dwn_signal_id: str = signal.out_lane_to_signal_id[lane_id]
            if dwn_signal_id is not None:
                lane = signal.signals[dwn_signal_id].observation.get_lane(lane_id)
                exiting_queued += lane.queued

        pressure_ = entering_queued - exiting_queued
        rewards[signal_id] = -pressure_
    return rewards


def queue_maxwait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        reward: float = 0.0
        for lane in signals[signal_id].observation.lanes.values():
            reward += lane.queued
            reward += lane.max_wait * cfg.coef
        rewards[signal_id] = -reward
    return rewards


def queue_maxwait_neighborhood(signals) -> dict[str, float]:
    rewards: dict[str, float] = queue_maxwait(signals)
    neighborhood_rewards: dict[str, float] = dict()
    for signal_id in signals:
        down_streams: dict[str, str] = cfg[signal_id]["downstream"]
        sum_reward: float = rewards[signal_id]
        for key in down_streams:
            neighbor = down_streams[key]
            if neighbor is not None:
                sum_reward += cfg.coop_gamma * rewards[neighbor]
        neighborhood_rewards[signal_id] = sum_reward
    return neighborhood_rewards


def fma2c(signals) -> dict[str, float]:
    region_fringes: dict[str, list[str]] = dict()
    fringe_arrivals: dict[str, int] = dict()
    liquidity: dict[str, int] = dict()
    for manager in cfg.management:
        region_fringes[manager] = list()
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is None
                or cfg.supervisors[neighbor] != cfg.supervisors[signal_id]
            ):
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = cfg.supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = cfg.supervisors[signal_id]
        fringes = region_fringes[manager]
        liquidity[manager] += signal.observation.departed - signal.observation.arrived
        for lane_id in signal.lanes:
            if lane_id in fringes:
                lane = signal.observation.get_lane(lane_id)
                fringe_arrivals[manager] = lane.arrived

    management_neighborhood: dict[str, float] = dict()
    for manager in cfg.management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in cfg.management_neighbors[manager]:
            mgr_rew += cfg.alpha * (fringe_arrivals[neighbor] + liquidity[neighbor])
        management_neighborhood[manager] = mgr_rew

    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            reward += lane.queued
            reward += lane.max_wait * cfg.coef
        rewards[signal_id] = -reward

    neighborhood_rewards: dict[str, float] = dict()
    for signal_id in signals:
        sum_reward = rewards[signal_id]

        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                sum_reward += cfg.alpha * rewards[neighbor]
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards
