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

extra_settings = [
    "log_level:INFO",
    "log_console:True",
    "compress_results:True",
    "delete_logs:True",
]

maps = None
if "grid" in run_name:
    maps = ["grid1x1"]  # grid1x2, grid4x4, arterial4x4
elif "resco_single" in run_name:
    maps = ["cologne1", "ingolstadt1"]
elif "resco_corridor" in run_name:
    maps = ["cologne3", "ingolstadt7"]
elif "resco_region" in run_name:
    maps = ["cologne8", "ingolstadt21"]
elif "germany" in run_name:
    maps = [
        "cologne1",
        "cologne3",
        "cologne8",
        "ingolstadt1",
        "ingolstadt7",
        "ingolstadt21",
    ]
elif "saltlake_peak" in run_name:
    maps = [
        "saltlake2_stateXuniversity run_peak:peak",
        "saltlake2_stateXuniversity controlled_signals:['A3'] run_peak:peak",
        "saltlake2_stateXuniversity controlled_signals:['B3'] run_peak:peak",
        "saltlake1A_stateXuniversity run_peak:peak",
        "saltlake1B_stateXuniversity run_peak:peak",
        "saltlake2_400sX200w run_peak:peak",
        "saltlake2_400sX200w controlled_signals:['A3'] run_peak:peak",
        "saltlake2_400sX200w controlled_signals:['B3'] run_peak:peak",
        "saltlake1A_400sX200w run_peak:peak",
        "saltlake1B_400sX200w run_peak:peak",
    ]
elif "saltlake_year" in run_name:
    episodes = 365 * 24
    decay_period = 31 * 24 / episodes
    maps = [
        f"saltlake2_stateXuniversity episodes:{episodes} epsilon_end:0.02 epsilon_decay_period:{decay_period}",
        f"saltlake2_400sX200w episodes:{episodes} epsilon_end:0.02 epsilon_decay_period:{decay_period}",
    ]
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
        with open("{0}_commands.sh".format(run_name), "w") as f:
            for command in commands:
                f.write("sed -i '$d' resco.slurm\n")
                f.write('echo "' + command + '" | tee -a resco.slurm > log.txt\n')
                f.write("sbatch ./resco.slurm\n")
    if not cfg.tamu_launcher:
        pool = mp.Pool(processes=int(total_processes))
        for command in commands:
            pool.apply_async(_fn, args=(command,))
            time.sleep(1)  # Required for optuna to work w/many processes
        pool.close()
        pool.join()
