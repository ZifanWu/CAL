⚠️Caution! This branch is for an experimenting improvement for the Qc ensemble implementation which intends to accelerate the ensemble value computation. Modifications mainly take place in [./agent/model.py](./agent/model.py) and [./agent/cal.py](./agent/cal.py). After having validated the performance of this improvement, this branch will be merged to the main branch.

# CAL
Code accompanying the paper ["Off-Policy Primal-Dual Safe Reinforcement Learning"](https://openreview.net/forum?id=vy42bYs1Wo).

<div align="center"><img src="/img/cal_fig1.png" alt="CAL" width="600" /></div>

## Installing Dependences
### Safety-Gym

```shell
cd ./env/safety-gym/
pip install -e .
```

We follow the environment implementation in the [CVPO repo](https://github.com/liuzuxin/cvpo-safe-rl/tree/main/envs/safety-gym) to accelerate the training process. All the compared baselines in the paper are also evaluated on this environment. For further description about the environment implementation, please refer to Appendix B.2 in the [CVPO paper](https://arxiv.org/abs/2201.11927).

### MuJoCo

Refer to https://github.com/openai/mujoco-py.



## Usage
Configurations for experiments, environments, and algorithmic components as well as hyperparameters can be found in [`/arguments.py`](/arguments.py).

### Training

For Safety-Gym tasks:

```shell
python main.py --env_name Safexp-PointButton1-v0 --num_epoch 500
```

For MuJoCo tasks:
```shell
python main.py --env_name Ant-v3 --num_epoch 300 --c 100 --qc_ens_size 8
```



### Settings for hyperparameter & algorithmic components

####  Safety-Gym

We adopt the same hyperparameter setting across all Safety-Gym tasks tested in our work (PointButton1, PointButton2, CarButton1, CarButton2, PointPush1), which is the default setting in [`/arguments.py`](/arguments.py). 

####  MuJoCo

The configurations different from the default setting are as follows:

- The conservatism parameter $k$ (`--k` in [`/arguments.py`](/arguments.py)) is 0. for Humanoid and 1. for HalfCheetah.

- The convexity parameter $c$ (`--c`) is 100 for Ant, 1000 for HalfCheetah and Humanoid.

- The replay ratio (`--num_train_repeat`) is 20 for HalfCheetah.

- The ensemble size $E$ of the safety critic (`--qc_ens_size`) is 8 for Ant where the task is harder and thus needs a larger ensemble.

  (In my test runs, the size of the ensemble does not significantly affect the running speed, thanks to the batch matrix multiplication function provided by PyTorch.)

> NOTE: While in CAL conservatism is originally incorporated in policy optimization, for the task of Humanoid we found it more effective to instead incorporate conservatism into $Q_c$ learning. To turn on this component, just set the option `--intrgt_max` to True.

### Logging
The codebase contains [wandb](https://wandb.ai/) as a visualization tool for experimental management. The user can initiate a wandb experiment by adding `--use_wandb` in the command above and specifying the wandb user account by `--user_name [your account]`.