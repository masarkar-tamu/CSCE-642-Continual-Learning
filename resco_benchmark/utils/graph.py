import os
import json
import numpy as np
import threading

import matplotlib
import matplotlib.pyplot as plt

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.logs import parse_logs

try:
    matplotlib.use("TkAgg")
    plot = True
except ImportError:
    print('Matplotlib  "TkAgg" not installed. Skipping graph.')
    plot = False


def parse_directory(log_dir):
    bot_lvl_dirs = list()
    for item in os.walk(log_dir):
        if len(item[2]) <= 3:
            continue  # Execution failed, skip empty results
        if "git_diff.txt" in item[2]:
            bot_lvl_dirs.append(item[0])
    print(bot_lvl_dirs)

    # Parse logs in parallel
    threads = list()
    for path in bot_lvl_dirs:
        threads.append(threading.Thread(target=parse_logs, args=(path, False)))
        threads[-1].start()
    for thread in threads:
        thread.join()


def combine_results(log_files):
    results = dict()
    better_name_collisions = dict()
    for log_file in log_files:
        fp = os.path.join(graph_dir, log_file)
        try:
            with open(fp, "r") as f:
                tmp = json.load(f)
        except json.decoder.JSONDecodeError:
            print("Skipping log file, not valid json:", fp)
            continue
        for metric in tmp:
            if metric not in results:
                results[metric] = dict()
            for log in tmp[metric]:
                if log not in cfg.names_map:
                    try:
                        better_name = log_file.split("/")[-1].split(".")[0]
                        better_name = better_name[:-17]
                    except:
                        better_name = log
                else:
                    better_name = log

                if better_name in better_name_collisions:
                    if better_name_collisions[better_name] != log:
                        better_name = log
                else:
                    better_name_collisions[better_name] = log

                if better_name in results[metric]:
                    results[metric][better_name] += tmp[metric][log]
                else:
                    if "timeLoss" in metric:
                        print("{0}: {1}".format(log, better_name))
                    results[metric][better_name] = tmp[metric][log]
    return results


def stack_trials(results, metric, exp_name, truncated=True):
    stack = list()
    minlen = np.inf
    for log in results[metric][exp_name]:
        if len(log) < minlen:
            minlen = len(log)
        stack.append(log)

    if truncated:
        for i in range(len(stack)):
            stack[i] = np.asarray(stack[i])[:minlen]
            # stack[i] = np.asarray(list(stack[i]) + [np.mean(stack)] * (cfg.episodes - len(stack[i])))
    else:
        raise NotImplementedError()  # Padding?

    return np.stack(stack)


def moving_average_filter(data, window):
    return np.convolve(data, np.ones(window), "valid") / window


def graph_it(results):
    # font = {'size': 22}
    # matplotlib.rc('font', **font)

    for metric in results.keys():
        if metric not in ["timeLoss", "vehicles"]: continue   # TODO
        map_to_plt = dict()
        for exp_name in results[metric].keys():
            stack = stack_trials(results, metric, exp_name)
            if exp_name not in cfg.names_map:
                if cfg.skip_unnamed:
                    print("Skipping exp, no name assigned:", exp_name)
                    continue
                else:
                    pretty_name = exp_name
                    try:
                        splitted = exp_name.split("+")
                        map_name = splitted[0]
                        pretty_name = " ".join(splitted[1:])
                        pretty_name = pretty_name.replace("_", " ")
                        pretty_name = pretty_name.title()
                        pretty_name = pretty_name.replace("@", ":")
                    except:
                        map_name = "null"
            else:
                splitted = cfg.names_map[exp_name].split("+")
                map_name = splitted[0]
                pretty_name = " ".join(splitted[1:])

            try_split = map_name.split("_")
            if len(try_split) > 1:
                map_name = try_split[0]
                mod = try_split[1]
                for modifier in cfg.names_map.modifiers.keys():
                    if mod == modifier:
                        mod = cfg.names_map.modifiers[modifier]

                if map_name in cfg.names_map.keys():
                    map_name = cfg.names_map[map_name]
                map_name = map_name + " " + mod
            else:
                if map_name in cfg.names_map.keys():
                    map_name = cfg.names_map[map_name]

            # Simple find-replace names
            for word in cfg.names_map.findreplace:
                pretty_name = pretty_name.replace(
                    word, cfg.names_map.findreplace[word]
                )

            stack = np.tile(stack, reps=10)
            print(stack.shape)

            exp_avg = np.mean(stack, axis=0)
            exp_std = np.std(stack, axis=0)
            exp_min = np.min(stack, axis=0)
            exp_max = np.max(stack, axis=0)
            for i in range(
                exp_avg.shape[0]
            ):  # TODO fill in gaps in data, not graph
                if exp_avg[i] == 0.0:
                    exp_avg[i] = exp_avg[i - 24]
                if exp_min[i] == 0.0:
                    exp_min[i] = exp_min[i - 24]
                if exp_max[i] == 0.0:
                    exp_max[i] = exp_max[i - 24]
            if "SLC" in map_name and len(exp_avg) > 200:
                print("replacing")
                exp_avg = np.asarray(list(exp_avg[:3880]) + list(exp_avg[3974:]))
                exp_std = np.asarray(list(exp_std[:3880]) + list(exp_std[3974:]))
                exp_min = np.asarray(list(exp_min[:3880]) + list(exp_min[3974:]))
                exp_max = np.asarray(list(exp_max[:3880]) + list(exp_max[3974:]))

            if cfg.smoothing is not None and stack.shape[1] >= cfg.smoothing:
                exp_avg = moving_average_filter(exp_avg, cfg.smoothing)
                exp_std = moving_average_filter(exp_std, cfg.smoothing)
                exp_min = moving_average_filter(exp_min, cfg.smoothing)
                exp_max = moving_average_filter(exp_max, cfg.smoothing)

            x = [i + 1 for i, _ in enumerate(range(stack.shape[1]))]

            if len(exp_avg) > 0:
                print(pretty_name, np.mean(exp_avg[-10:]))
                if map_name not in map_to_plt:
                    map_to_plt[map_name] = list()
                map_to_plt[map_name].append(
                    (pretty_name, x, exp_avg, exp_std, exp_min, exp_max)
                )

        plots = list()
        for map_name in map_to_plt:
            fig, ax = plt.subplots()
            fig.set_size_inches(16, 10, forward=True)
            min_y, max_y = np.inf, -np.inf
            algorithm_results = map_to_plt[map_name]
            algorithm_results.sort(key=lambda x: x[0:3])

            # TODO defines a custom order
            # custom_order = [
            #                     algorithm_results[3],
            #                     algorithm_results[4],
            #                     # algorithm_results[5],
            #                     algorithm_results[0],
            #                     algorithm_results[2],
            #                     algorithm_results[1],
            #                     # algorithm_results[4],
            #                 ]
            # algorithm_results = custom_order

            for name, x, y, z, a, b in algorithm_results:
                min_y = min(min_y, np.min(y))
                max_y = max(max_y, np.max(y))
                ax.plot(x, y, label=name)
                if cfg.error_bars:
                    if cfg.min_max:
                        ax.fill_between(x, a, b, alpha=0.4)
                    else:
                        ax.fill_between(x, y - z, y + z, alpha=0.4)

            title = cfg.names_map[metric] if metric in cfg.names_map else metric
            ylabel = (
                cfg.names_map["yaxis_" + metric]
                if "yaxis_" + metric in cfg.names_map
                else metric
            )

            ax.set_title("{0}: {1}".format(map_name, title))
            ax.set_xlabel("Hour")
            ax.set_ylabel(ylabel)

            ax.set_xlim(0, 100)

            # ax.set_ylim(1, 600)
            # ax.set_xlim(0, 4000)
            # ax.set_xlim(720, 768)
            plt.legend()

            # if max_y - min_y > 2000:  # TODO experimental rescaling for magnitude differences in plots
            #     ax.set_yscale('symlog')
            #     ax.autoscale_view()
            #     labels = list(np.round(np.arange(min_y, max_y, step=0.1*(max_y - min_y)), 0).astype(int))
            #     if max_y < 0:
            #         min_y_resolution = list(np.round(np.arange(max_y, labels[-1], step=0.2*(labels[-1] - max_y)), 0).astype(int))
            #         min_y_resolution.pop(-1)
            #     else:   # TODO untested else
            #         min_y_resolution = list(np.round(np.arange(min_y, labels[-1], step=0.2*(labels[-1] - min_y)), 0).astype(int))
            #         min_y_resolution.pop(-1)
            #     labels += min_y_resolution
            #     ax.set_yticks(labels, labels)
            fig.tight_layout()
            file_name = "{0}_{1}.png".format(
                map_name.replace("-", ""), title
            ).replace(" ", "_")
            fig.savefig(os.path.join(graph_dir, file_name))

            plots.append(ax)

        if "no_plot" not in cfg or cfg.no_plot is False:
            plt.show()


if __name__ == "__main__":
    graph_dir = cfg.log_dir
    parse_directory(graph_dir)

    included_logs = list()
    for log in list(os.listdir(graph_dir)):
        if ".json" in log:
            print("Found", log)
            included_logs.append(log)

    graph_it(combine_results(included_logs))
