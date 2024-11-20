====================
Installation
====================


Retrieve the source code from the `GitHub repository <https://github.com/Pi-Star-Lab/RESCO.git>`_.

Python's package manager, `pip <https://pip.pypa.io/en/stable/>`_, is used to install RESCO. The following will install
RESCO in editable mode. The `Simulation of Urban Mobility <https://eclipse.dev/sumo/>`_ (SUMO) will be installed via pip as well.

.. code-block:: bash

    cd RESCO
    pip install -e .

Optionally install the dependencies for hyper-parameter tuning [optuna], building this documentation [docs], or
executing the FMA2C/MA2C algorithms [fma2c]. The option [fma2c] requires a compatible python version for the outdated
version of tensorflow used, suggested is version 3.6.

.. code-block:: bash

    pip install -e .[optuna,docs,fma2c]

Installation isolation is recommended via
`miniconda <https://docs.anaconda.com/miniconda/#quick-command-line-install>`_. After installing miniconda, create a new
environment and install the dependencies using pip with the following commands.

.. code-block:: bash

    conda create -n resco python=3.12
    conda activate resco
    python -m pip install -e .[optuna,docs]


------------------
Nvidia GPU Support
------------------
If you want to use the GPU for the neural network based algorithms and you have a current version of CUDA installed you
do not need to install anything else. If you do not have a current version of CUDA installed you will need to reference
the `torch installation instructions <https://pytorch.org/get-started/locally/>`_ for your CUDA version. An example is
given below for CUDA 11.X.


.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cu118


--------------------
Compatible Platforms
--------------------

Please update this documentation with confirmed installation platforms. Please open an issue if you encounter any
deviation from these instructions so that they can be updated if necessary! The oldest confirmed SUMO version is listed
in each cell.


+----------------+-----+-----+-----+-----+------+------+-------------+
| Python:        | 3.6 | 3.7 | 3.8 | 3.9 | 3.10 | 3.11 | 3.12        |
+================+=====+=====+=====+=====+======+======+=============+
| Mint 21.3      |     |     |     |     |      |      | 1.20.0      |
+----------------+-----+-----+-----+-----+------+------+-------------+
| Ubuntu 24.04   |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
| Ubuntu 22.04.4 |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
| Ubuntu 20.04.6 |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
| Windows 11     |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
| Windows 10     |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
| Mac OS 14      |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
|                |     |     |     |     |      |      |             |
+----------------+-----+-----+-----+-----+------+------+-------------+
