from resco_benchmark.experiment_runner.common import *

algs = ["FIXED", "IDQN", "MPLight", "AdvancedMPLight"]

if "FMA2C" in run_name:  # FMA2C can only run in old python
    algs = ["FMA2C"]

extra_settings.append("converged:null")  # Run a fixed number of episodes

commands = []
for map_name in maps:
    for alg in algs:
        for _ in range(total_processes):

            if alg == "FIXED":
                obj = "fixed episodes:2 optuna_trials:" + str(
                    cfg.optuna_trials * cfg.episodes / 2
                )
            else:
                obj = "learning_rate"

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + alg,
                    "optuna_objective:" + obj,
                ]
                + extra_settings
            )

            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
