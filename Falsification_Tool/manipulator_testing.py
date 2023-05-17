from model.skrl_oige_model import skrl_oige_model
from monitor.stl_dense_offline import stl_dense_offline_monitor
from optimizer.optimizer import Optimizer

import os

if __name__ == "__main__":
 

    # Task choice: PointReaching, PegInHole, DoorOpen,
    # BallBalancing, BallPushing, BallCatching
    # CubeStacking, ClothPlacing
    task_name = "FrankaBallBalancing"
    
    # agent
    agent_type = "PPO"  # TRPO, PPO
    omniisaacgymenvs_path = os.path.realpath(
        os.path.join(os.path.realpath(__file__), "../../Gym_Envs")
    )
    agent_path = (
        omniisaacgymenvs_path
        + "/Final_Policy/BallBalancing/BallBalancing_skrl_"
        + agent_type
        + "/checkpoints/best_agent.pt"
    )
    
    # config
    simulation_max_steps = 100
    num_envs = 1
    opt_types = [
        # "random",
        "NelderMead",
        # "DualAnnealing",
    ]  # random, NelderMead, DualAnnealing
    global_budget = 4
    local_budget = 300

    # Load model under test (drl agent + oige env)
    test_model = skrl_oige_model(
        agent_path=agent_path,
        agent_type=agent_type,
        task_name=task_name,
        num_envs=num_envs,
        timesteps=simulation_max_steps,
    )


    for opt_type in opt_types:

        # Load STL monitor based on task
        monitor = stl_dense_offline_monitor(task_name=task_name, agent_type=agent_type)

        # global search
        for i in range(global_budget):

            print("Global trial: " + str(i))
            # Create optimizer
            optimizer = Optimizer(
                task_name,
                test_model,
                monitor,
                opt_type=opt_type,
                budget_size=local_budget,
            )

            # local search
            results = optimizer.optimize()
            
            print(results)

    # close simulation environment
    test_model.close_env()
