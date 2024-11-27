from resco_benchmark.experiment_runner.common import *

algs = ["FIXED", "MAXWAVE", "MAXPRESSURE", "IDQN", "MPLight", "AdvancedMPLight", "IPPO"]
# algs = ["FMA2C"]  # Requires old python

if "FMA2C" in run_name:  # FMA2C can only run in old python
    algs = ["FMA2C"]
    if "resco" in run_name:
        extra_settings.append("episodes:1400")


commands = []
for map_name in maps:
    for alg in algs:
        for _ in range(cfg.trials):
            if "IPPO" in alg:
                alg = "IPPO episodes:1400"

            cmd = " ".join(
                [python_cmd, "main.py", "@" + map_name, "@" + alg] + extra_settings
            )

            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
