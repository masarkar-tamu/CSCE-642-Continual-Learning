from resco_benchmark.experiment_runner.common import *

step_offs = [0, 1, 2, 4, 7, 10, 15]

commands = []
fixed_commands = []
for trial in range(cfg.trials):
    for map_name in maps:
        for step in step_offs:
            commands.append(
                " ".join(
                    [python_cmd, "main.py", "@" + map_name, "@FIXED cycle_forward:True ", "step_off:" + str(step)]
                    + extra_settings
                )
            )

            commands.append(
                " ".join(
                    [python_cmd, "main.py", "@" + map_name, "@FIXED", "step_off:" + str(step)]
                    + extra_settings
                )
            )

if __name__ == "__main__":
    launch_command(commands)
