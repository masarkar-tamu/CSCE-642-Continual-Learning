from resco_benchmark.experiment_runner.common import *

# TODO optionally build this from results directory
trained_models = {
    # Oracle+depart
    "04af47b7-cba5-4e67-a01b-a38013fd42d9": "045_epsilon",
    "13182009-458a-471e-9547-7637ed1cf085": "045_epsilon",
    "b16011f0-2339-4847-8317-50e6d36e03e6": "055_epsilon",
    "d447d11e-90c3-4770-8e56-18dfcf8cb3f0": "055_epsilon",
    "55b91200-0f4c-4b00-932c-4f445188441b": "075_epsilon",
    "b914f795-62ca-45ca-9d74-4bfb64c8f174": "075_epsilon",
    "49ec7c7f-8dd3-4913-8d4e-fea80fbc6edc": "95_epsilon",
    "081e9487-8fa7-4b07-9401-a916cffece28": "95_epsilon",
    "9c4b2c1c-60cb-4ffe-9ef8-7fdf11d3169d": "0995_epsilon",
    "b0502359-2bc1-4306-8b19-3d7b35a34362": "0995_epsilon",
}

# Overwrite common.extra_settings
extra_settings = ["log_console:True", "run_peak:peak"]

commands = []
for model in trained_models:
    commands.append(
        " ".join(
            [
                python_cmd,
                "main.py",
                "@saltlake2_stateXuniversity",
                "@IDQN",
                "explorer:fixed_explorer",
                "load_model:" + model,
                "model_name:" + trained_models[model],
                "training:False",
                "load_replay:False",
                "episodes:20",
                "converged:null",
            ]
            + extra_settings
        )
    )

if __name__ == "__main__":
    launch_command(commands)
