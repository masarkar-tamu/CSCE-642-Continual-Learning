from resco_benchmark.experiment_runner.common import *

rewards = [
    "maxqueue_maxdelay_div",
    "delay",
    "wait",
    "queue",
    "pressure",
    "maxqueue",
    "maxwait",
    "throughput",
    "oracle_delay",
    "oracle_delay_depart",
    "speed",
    "acceleration",
    "maxdelay",
]

commands = []
fixed_commands = []
for trial in range(cfg.trials):
    for map_name in maps:
        for reward in rewards:
            commands.append(
                " ".join(
                    [python_cmd, "main.py", "@" + map_name, "@IDQN", "reward:" + reward]
                    + extra_settings
                )
            )
            fixed_commands.append(
                " ".join(
                    [
                        python_cmd,
                        "main.py",
                        "@" + map_name,
                        "@FIXED",
                        "reward:" + reward,
                    ]
                    + extra_settings
                )
            )

commands = commands + fixed_commands
if __name__ == "__main__":
    launch_command(commands)
