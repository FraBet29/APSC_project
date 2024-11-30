# Mesh-Informed Bayesian Neural Networks
## Advanced Programming for Scientific Computing (APSC)

### Description

This project aims to develop a Bayesian extension of the DLROMs package for reduced order modeling of partial differential equations. In particular, we focus on combining the Bayesian approach with mesh-informed neural networks (MINNs), a class of models designed to handle functional data defined on a mesh. We test the implementation on two benchmark problems: the Darcy flow and the KPP-Fisher equation. The project is developed within the course Advanced Programming for Scientific Computing (APSC) at Politecnico di Milano.

### Code structure

The code is structured as follows:

```
.
├── src
│   └── dlroms_bayesian
│       ├── __init__.py
│       ├── bayesian.py
│       ├── expansions.py
│       ├── svgd.py
│       └── utils.py
├── config
├── experiments
│   ├── darcy_flow
│   └── brain_damage_recovery
└── tests
```

```src``` contains the source code of the project. The ```dlroms_bayesian``` module contains the implementation of the Bayesian extension of the DLROMs package. The ```config``` folder contains the configuration files for the experiments. The ```experiments``` folder contains subfolders for each benchmark. The ```tests``` folder contains some simple unit tests.

Each experiment folder has the following structure:

```
.
├── snapshots
├── results
├── results_bayesian
├── checkpoints
├── <experiment_name>_snapshots.py
├── <experiment_name>.py
├── <experiment_name>_bayesian.py
├── <experiment_name>_evaluate.py
└── <experiment_name>_bayesian_evaluate.py
```

```snapshots``` contains the snapshots used for training. ```results``` and ```results_bayesian``` contain plots and other results of the experiments. ```checkpoints``` contains the trained models. The ```<experiment_name>.py``` and ```<experiment_name>_bayesian.py``` files contain the training scripts for the DLROMs and the Bayesian DLROMs, respectively. The ```<experiment_name>_evaluate.py``` and ```<experiment_name>_bayesian_evaluate.py``` files contain the evaluation scripts for the DLROMs and the Bayesian DLROMs, respectively.

### Installation

It is recommended to create a new ```conda``` environment to run the code (refer to [this webpage](https://docs.anaconda.com/miniconda/install/) for Miniconda). You can create a new environment with the following command (which automatically installs the ```pip``` package manager):

```bash
conda create -n dlroms-env python=3.12
```

To activate the environment, type:

```bash
conda activate dlroms-env
```

Although FEniCS is listed as an optional dependency for DLROMs, it is needed to run the experiments. You can install it via:

```bash
conda install -c conda-forge fenics
```

Then, install the DLROMs package:

```bash
pip install git+https://github.com/NicolaRFranco/dlroms.git
```

Finally, install the Bayesian extension (from the root folder):

```bash
pip install .
```

### Usage

You can run all the experiments by executing the following command:

```bash
python run_experiments.py
```

This script will run all the experiments defined in the ```config``` folder. You can also run a single experiment by specifying the experiment name:

```bash
python run_experiments.py --experiment <experiment_name>
```

For both experiments, some evaluation metrics (relative test error, coverage, etc.) are printed on the terminal.

> _NOTE:_ The brain damage recovery experiment runs despite a Gmsh error displayed on the terminal.

To run the unit tests, execute the following command:

```bash
python run_tests.py
```

You should expect all tests to pass.
