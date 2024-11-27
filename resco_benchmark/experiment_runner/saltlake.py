from resco_benchmark.experiment_runner.common import *

algs = [
    "FIXED",
    "MAXWAVE",
    "MAXPRESSURE",
    "IDQN",
    "MPLight",
    "AdvancedMPLight",
    "IPPO",
    "ICoInPlus",
]
# algs = ["FMA2C"]   # Requires old python

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for alg in algs:
            episodes = ""  # Defaults to cfg.episodes
            if ("IPPO" in alg or "FMA2C" in alg) and "run_peak:peak" in map_name:
                episodes = f"episodes:{cfg.episodes*10}"

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + alg,
                    episodes,
                ]
                + extra_settings
            )
            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
