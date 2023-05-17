# Towards Understanding and Developing Trustworthy AI-CPS: Benchmarking and Testing AI-CPS in Robotics Manipulation with NVIDIA Omniverse Isaac Sim
This folder contains all revelant code for the paper "Towards Understanding and Developing Trustworthy AI-CPS: Benchmarking and Testing AI-CPS in Robotics Manipulation with NVIDIA Omniverse Isaac Sim".

## Benchmark of Robotics Manipulation 

### Requirements:
1. Install Omniverse Isaac Sim: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html
2. Add Isaac Sim to PYTHON_PATH (with default installation location of ISAAC SIM)
   ```
   alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
   ```
    
2. Install Omniverse Isaac GYM Envs: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs
3. Install SKRL, RTAMT, and Scipy in the Isaac Sim Python environment (the latter two are used for falsification): go to the Isaac folder, and run
   ```
   ./python.sh -m pip install skrl
   ./python.sh -m pip install rtamt
   ./python.sh -m pip install scipy
   ```
   
### Run the learning process:

To run SKRL with provided task environments (example):
```
cd Gym_Envs/
PYTHON_PATH skrl_train_PPO.py task=FrankaBallBalancing num_envs=16 headless=False
```

To launch Tensorboard: 
```
PYTHON_PATH -m tensorboard.main --logdir runs/FrankaBallBalancing/summaries/
```

## Falsification Tool
To run the falsification test for pre-trained agent, run:
```
cd Falsification_Tool/
PYTHON_PATH manipulator_testing.py headless=False
```

## Performance Evaluation
The performance evaluation uses the same framework as the falsification tool, but with the optimizer set to "random":
```
cd Eval/
PYTHON_PATH manipulator_eval.py headless=False
```
