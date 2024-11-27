from resco_benchmark.experiment_runner.common import *

commands = []
for map_name in maps:

    for __ in range(cfg.trials):
        cmd = " ".join(
            [
                python_cmd,
                "main.py",
                "@" + map_name,
                "@IDQN",
            ]
            + extra_settings
        )
        commands.append(cmd)

        cmd = " ".join(
            [
                python_cmd,
                "main.py",
                "@" + map_name,
                "@IDQN",
                "state:fixed_state",
                "reward:fixed_reward",
                "fixed_absolute:True",
            ]
            + extra_settings
        )
        commands.append(cmd)

        cmd = " ".join(
            [
                python_cmd,
                "main.py",
                "@" + map_name,
                "@IDQN",
                "state:fixed_state",
                "reward:fixed_reward",
                "fixed_relative:True",
            ]
            + extra_settings
        )
        commands.append(cmd)

        # cmd = ' '.join([python_cmd, 'main.py',
        #                 '@' + map_name,
        #                 '@IDQN',
        #                 'state:fixed_state',
        #                 'reward:fixed_shadow',
        #                 'fixed_reward_decay:2'
        #                 ] + extra_settings)
        # commands.append(cmd)
        #
        # cmd = ' '.join([python_cmd, 'main.py',
        #                 '@' + map_name,
        #                 '@IDQN',
        #                 'state:fixed_state',
        #                 'reward:fixed_shadow',
        #                 'fixed_reward_decay:3'
        #                 ] + extra_settings)
        # commands.append(cmd)

        cmd = " ".join(
            [
                python_cmd,
                "main.py",
                "@" + map_name,
                "@IDQN",
                "state:extended_state",
                "reward:fixed_reward",
                "fixed_absolute:True",
            ]
            + extra_settings
        )
        commands.append(cmd)

        cmd = " ".join(
            [
                python_cmd,
                "main.py",
                "@" + map_name,
                "@IDQN",
                "state:extended_state",
                "reward:fixed_reward",
                "fixed_relative:True",
            ]
            + extra_settings
        )
        commands.append(cmd)

        # cmd = ' '.join([python_cmd, 'main.py',
        #                 '@' + map_name,
        #                 '@IDQN',
        #                 'state:extended_state',
        #                 'reward:fixed_shadow',
        #                 'fixed_reward_decay:2'
        #                 ] + extra_settings)
        # commands.append(cmd)
        #
        # cmd = ' '.join([python_cmd, 'main.py',
        #                 '@' + map_name,
        #                 '@IDQN',
        #                 'state:extended_state',
        #                 'reward:fixed_shadow',
        #                 'fixed_reward_decay:3'
        #                 ] + extra_settings)
        # commands.append(cmd)


if __name__ == "__main__":
    launch_command(commands)
