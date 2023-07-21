import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from reservoir_env_ma import ReservoirEnv
from network_model_attention_cnn_reg_ma_no_emb import GTrXLNet as MyModel
from sim_opt_setup import Sim_opt_setup

import os

# parameters
num_cpus = 240 #int(sys.argv[1])
num_opt_ctrl_step = 19
num_sim_iter = 1
num_training_iter = 1000
memory_len = 5

# multi-Asset input
num_asset = 4
gen_sim_input = {}
for i in range(num_asset):
    gen_sim_input[f"res_{i+1}"] = Sim_opt_setup(f"res_{i+1}")
    gen_sim_input[f"res_{i+1}"]["reg_ctrl"] = True #False
    gen_sim_input[f"res_{i+1}"]["epl_mode"] = "negative_npv" 
    gen_sim_input[f"res_{i+1}"]["epl_start_iter"] = 100

cur_env_config = {"gen_sim_input": gen_sim_input}

ray.init(ignore_reinit_error=True, log_to_driver=False, address=os.environ["ip_head"], include_dashboard=False)#, memory=500 * 1024 * 1024)
ModelCatalog.register_custom_model("my_model", MyModel)

nstep = num_opt_ctrl_step*num_sim_iter

def on_train_result(info):
    result = info["result"]
    if result["training_iteration"] >= 100:
        #num_iter = result["training_iteration"]
        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_task.remote()))
                #lambda env: env.set_task()))

tune.run(
    "PPO",
    stop={ "training_iteration": num_training_iter,},
    config={
        "env": ReservoirEnv,
        "callbacks": {
            "on_train_result": on_train_result,
        },
        "model": {
            "custom_model": "my_model",
            "max_seq_len": memory_len,
            "custom_model_config": {
                "num_transformer_units": 2, #base 2
                "attention_dim": 128, #64
                "num_heads": 2, #base 2
                "memory_inference": memory_len, #num_opt_ctrl_step, 
                "memory_training": memory_len, #num_opt_ctrl_step,  
                "head_dim": 128, #64
                "position_wise_mlp_dim": 128,  #base 64
            },
        },
        
        "num_envs_per_worker": 2, # 4
        "remote_worker_envs": True,
        
        "num_workers": num_cpus,
        "num_cpus_for_driver": 12, #8,
        "num_gpus": 0,
        "train_batch_size": num_cpus * nstep,  # Total number of steps per iterations
       
        "batch_mode": "complete_episodes",
        #"rollout_fragment_length": nstep,
        
        "sgd_minibatch_size": 128, 
        
        "gamma": 0.9997,

        # "lr": 5e-5,
        # "entropy_coeff": 1e-4, #1e-3, 
        "lr_schedule": [[0, 1e-4], [num_cpus * nstep * num_training_iter, 1e-5]], 
		"entropy_coeff_schedule": [[0, 1e-3], [num_cpus * nstep * num_training_iter, 1e-5]],
        "vf_loss_coeff": 1,
        "num_sgd_iter": 10,
        
        
        "env_config": cur_env_config,
    },
    sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
    
   # checkpoint_at_end=True,
    checkpoint_freq = 5,
    local_dir="./logs", 
    restore="./logs/PPO/PPO_ReservoirEnv_96514_00000_0_2022-05-16_00-41-24/checkpoint_000165/checkpoint-165"
)

ray.shutdown()
