from resco_benchmark.experiment_runner.common import *

method = [
    "fasttrac:True",
    "crelu:True",
    "parameter_reset_freq:10",
    "parameter_reset_freq:24",
    "parameter_reset_freq:48",
    "buffer_size:6307200",
    "buffer_size:518400",
    "buffer_size:17280",
]

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for meth in method:

            cmd = " ".join(
                [python_cmd, "main.py", "@" + map_name, "@IDQN", meth] + extra_settings
            )
            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
