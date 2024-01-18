# CAL
Code accompanying the paper ["Off-Policy Primal-Dual Safe Reinforcement Learning"](https://openreview.net/forum?id=vy42bYs1Wo).

<div align="center"><img src="/img/cal_fig1.png" alt="CAL" width="500" /></div>

## Installing Dependences
### MuJoCo
Refer to https://github.com/openai/mujoco-py.

### Safety-Gym
> cd ./env/safety-gym/
> pip install -e .
We follow the environment implementation in the [CVPO](https://github.com/liuzuxin/cvpo-safe-rl/tree/main/envs/safety-gym) repo to accelerate the training process. All the compared baselines in the paper are also evaluated on this environment. For further description about the environment implementation, please refer to Appendix B.2 in the CVPO [paper](https://arxiv.org/abs/2201.11927).

## Usage
Configurations can be found in [`/arguments.py`](/arguments.py).

### Training
For MuJoCo tasks:
> python main_cal.py --env_name Hopper-v3 --num_epoch 300 --cuda_num 0

For Safety-Gym tasks:
> python main_cal.py --env_name Safexp-PointButton1-v0 --constraint_type safetygym --num_epoch 300 --cuda_num 0

### Hyperparameter Settings
The conservatism parameter $k$ (`--k` in [`/arguments.py`](/arguments.py)) is 0.5 for all tasks except for PointPush1 (0.8).
The convexity parameter $c$ (`--c` in [`/arguments.py`](/arguments.py)) is ​10 for all tasks except for Ant (100), HalfCheetah (1000) and Humanoid (1000).
The UTD ratio (`--num_train_repeat` in [`/arguments.py`](/arguments.py)) of CAL is 20 for all tasks except for Humanoid (10) and HalfCheetah (40).

### Logging
The codebase contains [wandb](https://wandb.ai/) as a visualization tool for experimental management. The user can initiate a wandb experiment by adding `--use_wandb` in the command above and specifying the wandb user account by `--user_name [your account]`.
