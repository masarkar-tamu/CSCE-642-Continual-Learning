from resco_benchmark.experiment_runner.common import *


# Used for RESCO scenarios where long-term data is unavailable
curriculum = [0.4, 0.6, 0.8, 1.0, 0.8, 0.6]
episodes = 20  # Move to next flow in curriculum every x episodes
iterations = 50  # Total number of passes over the curriculum


algs = ["FIXED", "IDQN", "MPLight", "AdvancedMPLight"]

b_size = "buffer_size:{0}".format(1000000)
dqn_variants = [b_size, b_size + " buffer_type:prioritized"]
for variant in dqn_variants:
    algs.append("IDQN " + variant)

if "FMA2C" in run_name:  # FMA2C can only run in old python
    algs = ["FMA2C"]


high = "flow:{0}".format(max(curriculum))
low = "flow:{0}".format(min(curriculum))
curriculum = curriculum * iterations
curric = 'curriculum:"' + str(curriculum) + '"'


decay_period = 0.8
if "saltlake" in run_name:
    # Find episodes in extra settings and remove it
    extra_settings = [x for x in extra_settings if not x.startswith("episodes:")]
    high = "run_peak:peak"
    low = "run_peak:low"
    curric = ""
    episodes = 365 * 24
    decay_period = 14 * 24 / episodes
elif maps is None:
    raise ValueError("Invalid run_name")


commands = []
for trial in range(cfg.trials):
    for map_name in maps:
        for alg in algs:
            algo = alg

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + algo,
                    "episodes:" + str(episodes),
                    curric,
                    # DQN specific parameters
                    "epsilon_end:0.05",  # Continual learning needs some exploration always
                    "epsilon_decay_period:"
                    + str(decay_period),  # Adjust schedule for longer period
                ]
                + extra_settings
            )
            commands.append(cmd)

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + algo,
                    high,
                ]
                + extra_settings
            )
            commands.append(cmd)

            cmd = " ".join(
                [python_cmd, "main.py", "@" + map_name, "@" + algo, low]
                + extra_settings
            )
            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
