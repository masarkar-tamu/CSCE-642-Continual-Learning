import sys
import os
import multiprocessing as mp
import subprocess
import shutil
import time
import uuid

# For example: saltlake_peak_global_sense, check setup in each runner, they are sometimes different
# If on a windows machine, first argument is not the path
if len(sys.argv) > 1:
    run_name = sys.argv[1]
    sys.argv.pop(1)  # Remove run_name so quik_config doesn't see it
else:
    raise EnvironmentError("You must provide a run_name as the first argument")

from resco_benchmark.config.config import config as cfg

# Reinsert run_name to argv at original position
sys.argv.insert(1, run_name)

if cfg.config_builder:
    cfg.trials = 1

if os.getcwd().endswith("experiment_runner"):
    os.chdir("..")
if os.getcwd().endswith("config"):
    os.chdir("..")

extra_settings = ["log_level:NOTSET", "log_console:True", "compress_results:True"]

if "_global_sense" in run_name:
    extra_settings.append("max_distance:20000")
if "_inc_flow" in run_name:
    extra_settings.append("flow:1.5")


maps = None
if "grid" in run_name:
    maps = ["grid1x1"]  # grid1x2, grid4x4, arterial4x4
elif "resco_single" in run_name:
    maps = ["cologne1", "ingolstadt1"]
elif "resco_corridor" in run_name:
    maps = ["cologne3", "ingolstadt7"]
elif "resco_region" in run_name:
    maps = ["cologne8", "ingolstadt21"]
elif "resco" in run_name:
    maps = [
        "cologne1",
        "cologne3",
        "cologne8",
        "ingolstadt1",
        "ingolstadt7",
        "ingolstadt21",
    ]
elif "saltlake" in run_name:
    maps = ["saltlake2_stateXuniversity"]  # , 'saltlake2_400sX200w']
    if "peak" in run_name or "inc_flow" in run_name or "high" in run_name:
        extra_settings.append("run_peak:peak")
    elif "low" in run_name:
        extra_settings.append("run_peak:low")
    else:
        extra_settings.append("episodes:" + str(24 * 365))
else:
    raise ValueError("Invalid run_name")


if cfg.processors is not None:
    total_processes = min(cfg.processors, mp.cpu_count())
else:
    total_processes = mp.cpu_count()

python_cmd = "python"
if shutil.which("python") is None:
    python_cmd = "python3"


def _fn(x):
    subprocess.call(x, shell=True)


def launch_command(commands):
    if cfg.tamu_launcher or cfg.log_level == "DEBUG" or cfg.log_level == "NOTSET":
        with open("{0}_commands_{1}.txt".format(run_name, uuid.uuid4()), "w") as f:
            for command in commands:
                f.write(command + "\n")
    if not cfg.tamu_launcher:
        pool = mp.Pool(processes=int(total_processes))
        for command in commands:
            pool.apply_async(_fn, args=(command,))
            time.sleep(1)  # Required for optuna to work w/many processes
        pool.close()
        pool.join()
