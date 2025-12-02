# Who Is Adam?

Basic code to get jax implementation of PPO running with Adam.

## Installation
1. Install anaconda or miniconda (https://www.anaconda.com/download/success)
2. create new conda python environment in terminal: conda create -n adam python=3.12
3. conda activate adam
4. pip install requirements.txt; cd optimize; pip install -e .
5. run optimize/experiments/gymnax/ppo/ppo/ppo_discrete.py
6. use wandb to see run progression, adjust config_ppo.yaml to adjust the config
7. run sweep_betas.sh to run sweeps of multiple values for momentum (beta_1)