# Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning

## Overview
This repository contains implementation of the preference-based multi-objective reinforcement learning (PBMORL).

## Code Structure
```
.
|--morl --> source codes for PBMORL
|--configs --> configs of environments
|--environments --> available environments of PBMORL
|--externals --> external algorithm package of PPO(https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).
```

## Requirements
- Python version: tested in Python 3.7.4
- Torch version: tested in Torch 1.10.2
- **MuJoCo** : install mujoco and mujoco-py by following the instructions in [mujoco-py](<https://github.com/openai/mujoco-py>).

## Getting Started
You can either install the dependencies in a conda virtual env (recomended) or manually. 

For conda virtual env installation, create a virtual env named **pbmorl** by:

```
conda env create -f PBMORL_environment.yml
```

## Run the code
```
conda activate pbmorl
cd morl
python run.py
```
The obtained policies are stored under the folders of morl/env_name/final

## Citation
