# Benchmarking Manipulation (with new version of Isaac Sim)
Benchmarking Manipulation Tasks with Isaac Sim 2022.2

Requirements:
1. Install ISAAC SIM: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html
2. Install ISAAC SIM GYM Envs: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs
3. Install SKRL in the ISAAC SIM PYTHON environment: go to default isaac folder, and run
```
./python.sh -m pip install skrl
```

First run (with default installation location of ISAAC SIM)
```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2022.2.0/python.sh
```


Or add the above line to `~/.bashrc`

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`:

```bash
PYTHON_PATH -m pip install -e .
```

To run skrl with envs
```
cd Gym_Envs/
PYTHON_PATH skrl_train.py task=FrankaBallPushing num_envs=16 headless=False
```

To run envs (FrankaBallPushing, FrankaBallBalancing, FrankaDoorOpen, FrankaPegInHole, FrankaClothFolding)
```
cd Gym_Envs/
PYTHON_PATH rl_train_multi.py task=FrankaBallPushing
```

Tensorboard: Tensorboard can be launched during training via the following command:
```
PYTHON_PATH -m tensorboard.main --logdir runs/FrankaBallBalancing/summaries/
```


To load a pre-trained checkpoint and run inferencing, run:
```
PYTHON_PATH rl_train_multi.py task=FrankaBallBalancing test=True checkpoint=runs/FrankaBallBalancing/nn/FrankaBallBalancing.pth num_envs=4
```

To test a pre-trained agent, run:
```
PYTHON_PATH manipulator_testing.py headless=True
```
