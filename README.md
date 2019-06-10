### Projected Proximal Policy Optimization

<a href="mailto:sergei.volodin@epfl.ch">Sergei Volodin</a>. <a href="http://epfl.ch">Swiss Federal Institute of Technology in Lausanne</a> (EPFL)

Course project for Theory and Methods of Reinforcement Learning, EE-618 at EPFL

## Agents
We consider CPO, sDQN, PPPO and a random agent. See our <a href="https://www.overleaf.com/read/cvxkswbspgpb">report</a> for more details

## How to run experiments
Tested on Ubuntu 16.04.5 LTS with 12 CPU, 60GB of RAM and 2x GPU NVidia GeForce 1080.

1. Install <a href="https://docs.conda.io/en/latest/miniconda.html">Anaconda</a> (Python 3.7 option)
3. Clone/download: `git clone https://github.com/sergeivolodin/SafeContinuousStateRL.git; cd SafeContinuousStateRL`
4. Install requirements: `conda env create -f environment.yml`
5. Run all settings by calling `run_all.sh`
6. It will produce `output/*.output` files and `output/figures/*.pdf` files, as well as will output run information to `run_*.txt`
7. Run the `analyze_run.ipynb` notebook to produce figures

## Project structure
1. `experiment.py` is the main file containing one experiment (loading agent, training, computing metrics)
2. `saferl.py` defines a `ConstrainedEnvironment` and the `ConstrainedAgent` abstract classes as well as helpers and the function to create a safe environment `make_safe_env`
3. `sppo.py` implements Projected Proximal Policy Optimization
4. `baselines.py` implements CPO and a random agent
5. `cartpole_safety_sdqn.ipynb` is the (non-working) implementation of sDQN
6. `config.py` contains the parameters of the experiment
7. `helpers.py` contains the functions for run analysis
8. `tf_helpers.py` contains some helper functions using TensorFlow
9. `costs.py` implements costs for environments
10. `cartpole_safety_a2c.ipynb` implements an (unsafe) A2C
11. `tfshow.py` embeds a TF graph into a Jupyter notebook, from <a href="https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter">StackOverflow</a>
12. `create_run.py` creates the `.sh` script from `config.py`
13. `analyze_run.ipynb` analyzes output produced by training (the `.sh` script) and writes output to `run_*.txt` and figures to `output/figures`
14. `output/*.sh` files consist of many lines of the form `python ../experiment.py --param1 v1 --param2 v2 ...`, running at most 16 processes in total (8 per GPU)
15. `output/*.output` files contain outputs of `experiment.py` (one run corresponds to one file)
16. `output/figures` contains generated figures
17. `run_setting.sh` runs a particular setting (create + `.sh` + analyze) and writes data to a file
18. `run_all.sh` runs all settings
19. Other files are not used
