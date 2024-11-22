from __future__ import annotations
import typing

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.traffic_signal import Signal, Lane


def advanced_mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.current_phase]
        for direction in signal.lane_sets:
            total_demand = 0
            inbound_queue_length = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                inbound_queue_length += lane.queued
                total_demand += lane.vehicle_count
            obs.append(total_demand)
            if len(signal.lane_sets[direction]) != 0:
                inbound_queue_length /= len(signal.lane_sets[direction])

            outbound_queue_length = 0
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    outbound_queue_length -= lane.queued
            if len(signal.lane_sets_outbound[direction]) != 0:
                outbound_queue_length /= len(signal.lane_sets_outbound[direction])
            obs.append(inbound_queue_length - outbound_queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def tabular(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        signal_obs = list()
        signal_obs.append(signal.current_phase)
        for direction in signal.lane_sets:
            total_wait = 0
            for lane_id in signal.lane_sets[direction]:
                for vehicle in signal.observation.get_lane(lane_id).vehicles.values():
                    total_wait += vehicle.wait
            signal_obs.append(np.rint(total_wait/5))
        observations[signal_id] = np.asarray(signal_obs)
    return observations


def fixed_state(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        n_acts = len(signal.green_phases)
        obs = [0] * n_acts
        obs[signal.current_phase] = 1
        for val in signal.time_since_phase.values():
            obs.append(val)
        observations[signal_id] = np.asarray(obs)
    return observations


def extended_state(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [[0.0] * 12]  # Extra space

        obs[0][-1] = signal.observation.arrived
        obs[0][-2] = signal.observation.departed
        for i in signal.time_since_phase:
            if i == signal.current_phase:
                obs[0][i] = 0
                obs[0][-3] = signal.time_since_phase[i]
            else:
                obs[0][i] = signal.time_since_phase[i]

        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            sig_lane_obs = signal.observation.get_lane(lane)

            lane_obs.append(sig_lane_obs.approaching)
            lane_obs.append(sig_lane_obs.queued)

            wait_sum, speed_sum, accel_sum, decel_sum, delay_sum = 0, 0, 0, 0, 0
            max_wait, max_speed, max_accel, max_decel, max_delay = 0, 0, 0, 0, 0
            for vehicle in sig_lane_obs.vehicles:
                vehicle = sig_lane_obs.vehicles[vehicle]
                wait_sum += vehicle.wait
                speed_sum += vehicle.average_speed
                delay_sum += vehicle.delay
                if vehicle.wait > max_wait:
                    max_wait = vehicle.wait
                if vehicle.average_speed > max_speed:
                    max_speed = vehicle.average_speed
                if vehicle.delay > max_delay:
                    max_delay = vehicle.delay

                accel = vehicle.acceleration
                if accel < 0:
                    decel = -1 * accel
                    decel_sum += decel
                    if decel > max_decel:
                        max_decel = decel
                elif accel > 0:
                    accel_sum += accel
                    if accel > max_accel:
                        max_accel = accel

            lane_vehicles_cnt = len(sig_lane_obs.vehicles)
            if lane_vehicles_cnt == 0:
                lane_vehicles_cnt = 1
            lane_obs.append(wait_sum / lane_vehicles_cnt)
            lane_obs.append(speed_sum / lane_vehicles_cnt)
            lane_obs.append(accel_sum / lane_vehicles_cnt)
            lane_obs.append(decel_sum / lane_vehicles_cnt)
            lane_obs.append(delay_sum / lane_vehicles_cnt)

            lane_obs.append(max_wait)
            lane_obs.append(max_speed)
            lane_obs.append(max_accel)
            lane_obs.append(max_decel)
            lane_obs.append(max_delay)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.current_phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.observation[lane]["approach"])
            lane_obs.append(signal.observation[lane]["total_wait"])
            lane_obs.append(signal.observation[lane]["queue"])

            total_speed = 0
            vehicles = signal.observation[lane]["vehicles"]
            for vehicle in vehicles:
                total_speed += vehicle["speed"]
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.current_phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                queue_length += lane.queued

            # Subtract downstream
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    queue_length -= lane.queued
            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.current_phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.observation[lane]["queue"]
                total_wait += signal.observation[lane]["total_wait"] / 28
                total_speed = 0
                vehicles = signal.observation[lane]["vehicles"]
                for vehicle in vehicles:
                    total_speed += vehicle["speed"]
                tot_approach += signal.observation[lane]["approach"] / 28

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].observation[lane][
                        "queue"
                    ]
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations


def wave(signals: dict[str, Signal]) -> dict[str, np.ndarray]:
    states: dict[str, typing.Any] = dict()
    for signal_id in signals:
        signal: Signal = signals[signal_id]
        state: np.ndarray = np.zeros(len(signal.lane_sets))

        for i, direction in enumerate(signal.lane_sets):
            for lane_id in signal.lane_sets[direction]:
                lane: Lane = signal.observation.get_lane(lane_id)
                state[i] += lane.queued + lane.approaching

        states[signal_id] = state
    return states


def ma2c(signals):
    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            waves.append(
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(cfg.coop_gamma * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.observation[lane]["max_wait"]
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    region_fringes = dict()
    for manager in cfg.management:
        region_fringes[manager] = []
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
                    region_fringes[mgr].append(inbounds)

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            lane_wave[lane_id] = lane.queued + lane.approaching

    manager_obs = dict()
    for manager in region_fringes:
        fringes = region_fringes[manager]
        waves = []
        for direction in fringes:
            summed = 0
            for lane_id in direction:
                summed += lane_wave[lane_id]
            waves.append(summed)
        manager_obs[manager] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in cfg.management_neighbors[manager]:
            neighborhood.append(cfg.alpha * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            waves.append(lane.queued + lane.approaching)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                waves.append(cfg.alpha * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            max_wait = lane.max_wait
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    region_fringes = dict()
    for manager in cfg.management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if (
                neighbor is None
                or cfg.supervisors[neighbor] != cfg.supervisors[signal_id]
            ):
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = cfg.supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = (
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in cfg.management_neighbors[manager]:
            neighborhood.append(cfg.alpha * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            waves.append(
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )

            waves.append(signal.observation[lane]["total_wait"] / 28)
            total_speed = 0
            vehicles = signal.observation[lane]["vehicles"]
            for vehicle in vehicles:
                total_speed += vehicle["speed"] / 20 / 28
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                waves.append(cfg.alpha * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.observation[lane]["max_wait"]
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations
