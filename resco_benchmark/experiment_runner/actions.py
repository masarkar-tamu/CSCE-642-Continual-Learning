from resco_benchmark.experiment_runner.common import *

actions = ["FixedCycle"]  # 'Phase' is default

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for action in actions:
            commands.append(
                " ".join(
                    [
                        python_cmd,
                        "main.py",
                        "@" + map_name,
                        "@IDQN",
                        "action_set:" + action,
                    ]
                    + extra_settings
                )
            )

launch_command(commands)
