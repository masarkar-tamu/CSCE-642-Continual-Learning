
How To Make Your Deep Reinforcement Learning Model More Robust For Traffic Signal Control?
==============================================
.. figure:: docs/source/_static/maps.png
  :alt: Six real-world traffic scenarios in RESCO.

|

This repo is a clone of private branch of the RESCO repository (https://github.com/Pi-Star-Lab/RESCO), maintained under Mr. James Ault. RESCO provides an interface for traffic network simulation and includes a set of algorithms for traffic signal control. This branch builds upon the RESCO codebase, leveraging insights from experiments conducted by Dr. Guni Sharon and Mr. James Ault, which validated the performance of DRL methods in non-stationary traffic scenarios. Images and experimental details featured in this project were also provided by Dr. Sharon and Mr. Ault.

**Install SUMO**

- Install SUMO: https://sumo.dlr.de/docs/Installing/index.html

**Install Relevant Python Packages**

.. code-block:: bash

    cd ~/resco_benchmark
    pip install -e ../

Our Implementation
------------

Inspired by the ability of the Continual Backpropagation (CBP) algorithm to enhance continual learning in deep neural networks, we explored its potential as described in the [original paper](https://arxiv.org/abs/2303.07507). We integrated the CBP algorithm into our codebase through the following steps:

1. **Adding CBP Layers**:
   We implemented two CBP layers, `cbp_conv` and `cbp_linear`, and organized them within the `cbp_layers` folder under the `agents` directory.

2. **Integrating with IDQN**:
   We applied the CBP layers to the IDQN method, which is implemented in the `pfrl_cbpdqn.py` file.

3. **Hyperparameter Settings**:
   For our experiments, we configured the following key hyperparameters:

   - Replacement rate: `0.0001`
   - Maturity threshold: `50,000`
   - Decay rate: `0.99`

How We Ran Our Experiments
------------

We conducted three experiments to evaluate different traffic signal control methods under various conditions:

**1. Comparison of Methods on a Single Intersection (800 hours)**: Tested CBP-IDQN, DQN, MPLight, Fixed-Time, and Max-Pressure methods over a simulated 800-hour period. Traffic flow significantly increased after 600 hours. Each method was run five times with different seeds to measure performance variance.

- Traffic environment setting before running:
    - Go to `config.yaml` and set episodes to 800.
    - Go to `agent.yaml` and set `epsilon_end` to 0 for `IDQN` and `CBPIDQN`.
    - Go to `multi_signal.py` on line `427` and set `cfg.flow` to 2.0 on condition `self.cummulative_episode` greater than 600.

- Then run the following commands:

.. code-block:: bash


    python main.py @ingolstadt1 @CBPIDQN
    python main.py @ingolstadt1 @IDQN
    python main.py @ingolstadt1 @MPLight
    python main.py @ingolstadt1 @FIXED
    python main.py @ingolstadt1 @MAXPRESSURE


**2. Comparison of Methods on Salt Lake Map (400 hours)**: Applied the same methods to a simulation using real-time data from the ATSPM website. Traffic flow significantly increased after 300 hours to simulate non-stationary conditions.

- Traffic environment setting before running: same as above.
- Then run the following commands:

.. code-block:: bash

    python main.py @saltlake2_stateXuniversity @CBPIDQN
    python main.py @saltlake2_stateXuniversity @IDQN
    python main.py @saltlake2_stateXuniversity @MPLight
    python main.py @saltlake2_stateXuniversity @FIXED
    python main.py @saltlake2_stateXuniversity @MAXPRESSURE

**3. CBP-IDQN vs. DQN (Detailed Analysis):** Focused solely on CBP-IDQN and DQN for a more detailed comparison. Epsilon end was set to 0.02 to encourage exploration and adaptability to dynamic traffic patterns.

- Traffic environment setting before running:
    - Go to `config.yaml` and set episodes to 400.
    - Go to `agent.yaml` and set `epsilon_end` to 0.02 for `IDQN` and `CBPIDQN`.
    - Go to `multi_signal.py` on line `427` and set `cfg.flow` to 2.0 on condition `self.cummulative_episode` greater than 300.

- Then run the following commands:

.. code-block:: bash

    python main.py @ingolstadt1 @CBPIDQN
    python main.py @ingolstadt1 @IDQN

References
------------

*Ault, James, and Guni Sharon. "Reinforcement Learning Benchmarks for Traffic Signal Control."*
**Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) Datasets and Benchmarks Track, December 2021.**
`Reinforcement Learning Benchmarks for Traffic Signal Control <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f0935e4cd5920aa6c7c996a5ee53a70f-Abstract-round1.html>`_

*S. Dohare, J. F. Hernandez-Garcia, Q. Lan, et al., “Loss of plasticity in deep continual learning,”*
vol. 632, pp. 768–774, 2024
