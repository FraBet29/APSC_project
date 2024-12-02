# Mesh-Informed Bayesian Neural Networks

## Description

This project aims to develop a Bayesian extension of the DLROMs package for reduced order modeling of partial differential equations. In particular, we focus on combining the Bayesian approach with mesh-informed neural networks (MINNs), a class of models designed to handle functional data defined on a mesh. We test the implementation on two benchmark problems: the Darcy flow and the KPP-Fisher equation. The project is developed within the course Advanced Programming for Scientific Computing (APSC) at Politecnico di Milano.

## Code structure

The directory is structured as follows:

```
.
├── src
│   └── dlroms_bayesian
├── config
├── experiments
│   ├── darcy_flow
│   └── brain_damage_recovery
├── tests
├── run_experiments.py
└── run_tests.py
```

```src``` contains the source code of the project. The ```dlroms_bayesian``` module contains the implementation of the Bayesian extension of the DLROMs package. The ```config``` folder contains the configuration files for the experiments. The ```experiments``` folder contains subfolders for each benchmark. The ```tests``` folder contains some simple unit tests. The ```run_experiments.py```  and ```run_tests.py``` scripts run the experiments and the tests, respectively (see [Usage](#usage) section).

The source code is composed of the following files:

```
.
└── dlroms_bayesian
    ├── __init__.py
    ├── bayesian.py
    ├── expansions.py
    ├── svgd.py
    └── utils.py
```

```bayesian.py``` contains the definition of the ```Bayesian``` and ```VariationalInference``` classes. ```svgd.py``` contains the definition of the ```SVGD``` class, implementing the Stein Variational Gradient Descent (SVGD) algorithm. ```expansions.py``` contains the extended version of the DLROMs ```Sparse``` layer with the deterministic and hybrid initialization strategies. ```utils.py``` contains helper functions.

Each experiment folder has the following structure:

```
.
├── snapshots
├── checkpoints
├── notebooks
├── <experiment_name>_snapshots.py
├── <experiment_name>.py
├── <experiment_name>_bayesian.py
├── <experiment_name>_evaluate.py
└── <experiment_name>_bayesian_evaluate.py
```

```snapshots``` contains the snapshots used for training and testing. ```checkpoints``` contains the weights of the trained models. The ```<experiment_name>_snapshots.py``` file can be used to generate the snapshots. The ```<experiment_name>.py``` and ```<experiment_name>_bayesian.py``` files contain the training scripts for the deterministic and Bayesian models, respectively. The ```<experiment_name>_evaluate.py``` and ```<experiment_name>_bayesian_evaluate.py``` files contain the evaluation scripts for the deterministic and Bayesian models, respectively. The ```notebooks``` folder contains Jupyter notebooks to train the models in a Google Colaboratory environment (see [Installation](#installation) section).

## Installation <a name="installation"></a>

### Google Colaboratory (optional)

The GitHub repository already contains some pre-trained model checkpoints. The training step was done using an NVIDIA Tesla T4 GPU provided by Google Colaboratory; the training time depends on the specific experiment configuration, typically ranging from 5 to 40 minutes. It is possible to train the models from scratch in the same environment using the Jupyter notebooks provided in the repository (```notebooks``` folders). To do so, the notebooks and the data must be uploaded to Google Drive. To install the required packages in Colaboratory, this cell can be added at the beginning of the notebook:

```python
from IPython.display import clear_output as clc

try:
    from dlroms import*
except:
    !pip install git+https://github.com/NicolaRFranco/dlroms.git
    from dlroms import*

try:
    from dlroms_bayesian import*
except:
    !pip install git+https://github.com/FraBet29/APSC_project.git
    from dlroms_bayesian import*

clc()
```

The DLROMs installation will automatically trigger the installation of FEniCS. 

If you have a GPU on your local machine, you can directly use the ```<experiment_name>.py``` and ```<experiment_name>_bayesian.py``` scripts to train the models. To set up the environment, see the next section.

### Local

To run the evaluation scripts locally (either on CPU or GPU), it is recommended to create a new ```conda``` environment (refer to [this webpage](https://docs.anaconda.com/miniconda/install/) for the installation of Miniconda) by running the following command (which automatically installs the ```pip``` package manager):

```bash
conda create -n <environment_name> python=3.12
```

To activate the environment:

```bash
conda activate <environment_name>
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

## Usage <a name="usage"></a>

You can run the evaluation scripts by typing (from the root folder):

```bash
python run_experiments.py
```

This script will run all the experiments defined in the ```config``` folder. You can also run a single experiment by specifying the experiment name:

```bash
python run_experiments.py --experiment <experiment_name>
```

For both experiments, some evaluation metrics (relative test error, coverage, etc.) are printed on the terminal.

> _NOTE: The brain damage recovery experiment runs despite a Gmsh error displayed on the terminal._

To run the unit tests, execute (from the root folder):

```bash
python run_tests.py
```

You should expect all tests to pass.
