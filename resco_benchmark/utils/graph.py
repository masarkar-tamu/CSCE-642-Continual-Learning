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
        if "config.json" in item[2]:
            bot_lvl_dirs.append(item[0])

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
                if len(tmp[metric][log][0]) < cfg.episodes:
                    print(log_file)
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
    if len(stack) < cfg.trials:
        print(exp_name, len(stack), minlen)
    return np.stack(stack)


def moving_average_filter(data, window):
    return np.convolve(data, np.ones(window), "valid") / window


def graph_it(results):
    font = {"size": 22}
    matplotlib.rc("font", **font)

    for stat in ["minmax", "std"]:
        for metric in results.keys():
            map_to_plt = dict()
            for exp_name in results[metric].keys():
                stack = stack_trials(results, metric, exp_name)
                if exp_name not in cfg.names_map:
                    pretty_name = exp_name
                    try:
                        splitted = exp_name.split("+")
                        map_name = splitted[0]
                        pretty_name = " ".join(splitted[1:])

                        if "controlled" in pretty_name:
                            map_name += splitted[1].split("_")[0]

                        # Simple find-replace names
                        for word in cfg.names_map.findreplace:
                            pretty_name = pretty_name.replace(
                                word, cfg.names_map.findreplace[word]
                            )

                        pretty_name = pretty_name.replace("_", " ")
                        pretty_name = pretty_name.replace("@", ":")
                    except:
                        map_name = "null"
                else:
                    splitted = cfg.names_map[exp_name].split("+")
                    map_name = splitted[0]
                    pretty_name = " ".join(splitted[1:])

                if map_name in cfg.names_map.keys():
                    map_name = cfg.names_map[map_name]

                exp_avg = np.mean(stack, axis=0)
                exp_std = np.std(stack, axis=0)
                exp_min = np.min(stack, axis=0)
                exp_max = np.max(stack, axis=0)

                if cfg.smoothing is not None and stack.shape[1] >= cfg.smoothing:
                    exp_avg = moving_average_filter(exp_avg, cfg.smoothing)
                    exp_std = moving_average_filter(exp_std, cfg.smoothing)
                    exp_min = moving_average_filter(exp_min, cfg.smoothing)
                    exp_max = moving_average_filter(exp_max, cfg.smoothing)

                x = [i + 1 for i, _ in enumerate(exp_avg)]

                if len(exp_avg) > 0:
                    table = (
                        " & ".join(
                            [
                                pretty_name,
                                str(np.mean(exp_avg[-10:])),
                                str(np.std(exp_avg[-10:])),
                            ]
                        )
                        + " \\\\"
                    )
                    # print(table)
                    if map_name not in map_to_plt:
                        map_to_plt[map_name] = list()
                    map_to_plt[map_name].append(
                        (pretty_name, x, exp_avg, exp_std, exp_min, exp_max)
                    )

            plots = list()
            for map_name in map_to_plt:
                fig, ax = plt.subplots()
                fig.set_size_inches(16, 10, forward=True)
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

                max_y = 0
                for name, x, y, z, a, b in algorithm_results:
                    ax.plot(x, y, label=name)
                    max_y = max(max_y, max(y[:10]))
                    if cfg.error_bars:
                        if stat == "minmax":
                            ax.fill_between(x, a, b, alpha=0.4)
                        else:
                            ax.fill_between(x, y - z, y + z, alpha=0.4)

                if "peak" in map_name.lower():
                    demand = "peak"
                    ax.set_xlabel("Episode")
                    ax.set_xlim(0, 100)
                else:
                    demand = "year"
                    ax.set_xlabel("Hour")
                ylabel = (
                    cfg.names_map["yaxis_" + metric]
                    if "yaxis_" + metric in cfg.names_map
                    else metric
                )

                # ax.set_xlim(720, 768)

                ax.set_title(f"{map_name}")
                ax.set_ylabel(ylabel)
                # ax.set_yscale("symlog")
                ax.set_ylim(max(0, ax.get_ylim()[0]), max_y * 10)
                plt.legend()
                fig.tight_layout()

                img_dir = str(
                    os.path.join(
                        cfg.log_dir,
                        "graphs",
                        metric.replace(" ", "_").lower(),
                        demand,
                        stat + "_smooth" + str(cfg.smoothing),
                    )
                )
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                file_name = "{0}.png".format(map_name.replace("-", "")).replace(
                    " ", "_"
                )
                fig.savefig(os.path.join(img_dir, file_name))

                plots.append(ax)

            if "plot" in cfg:
                plt.show()


if __name__ == "__main__":
    graph_dir = cfg.log_dir
    parse_directory(graph_dir)

    included_logs = list()
    for log in list(os.listdir(graph_dir)):
        if ".json" in log:
            included_logs.append(log)

    graph_it(combine_results(included_logs))
