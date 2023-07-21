import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

from reservoir_sim_ma import Simulator
import gc
import psutil
import copy

class ReservoirEnv(gym.Env):
    def __init__(self, env_config):
               
        self.gen_sim_input = copy.deepcopy(env_config["gen_sim_input"])
        self.sim_input = self.gen_sim_input["res_1"]

        # simulator
        self.res_sim = Simulator()
        
        # initialize history collector
        self.hist = self.History()
        
        # number of assets
        self.num_assets = len(self.gen_sim_input)
        
        # get total number of wells and dummy obs
        self.num_prod = 0
        self.num_inj = 0
        self.num_wells = 0
        self.max_avail_act = 0
        self.asset_obs = {}
        self.well_ids = {}
        for asset in self.gen_sim_input:
            num_prod, num_inj = self.gen_sim_input[asset]["num_prod"], self.gen_sim_input[asset]["num_inj"]
            self.num_prod += num_prod
            self.num_inj += num_inj
            self.max_avail_act = np.max([self.max_avail_act, num_prod+num_inj])
            
            # dummy obs for asset
            num_obs = 3 * num_prod + 2 * num_inj
            if self.gen_sim_input[asset]["reg_ctrl"]:
                num_obs += num_prod + num_inj
            self.asset_obs[asset] = np.zeros((self.gen_sim_input[asset]["num_run_per_step"], num_obs))
            
            # well IDs
            self.well_ids[asset] = np.arange(self.num_wells+1, self.num_wells+num_prod+num_inj+1)
            
            # total number of wells
            self.num_wells += num_prod + num_inj
 
        # pad well ids to add inactive wells
        for asset in self.gen_sim_input:
            self.well_ids[asset] = np.pad(self.well_ids[asset], \
                             (0, self.max_avail_act-self.well_ids[asset].shape[0]))
        
        
        # action and observation space 
        self._setup_spaces()
        
        # training realizations
        worker_ind = np.max([env_config.worker_index - 1, 0])
        num_cluster = np.max(self.sim_input["cluster_labels"]) + 1
        self.cluster_index = worker_ind % num_cluster  # zero-based

        self.realz_train = {}
        for asset in self.gen_sim_input:
            self.realz_train[asset] = np.argwhere(self.gen_sim_input[asset]["cluster_labels"] == self.cluster_index) + 1
            mask = np.in1d(self.realz_train[asset], self.gen_sim_input[asset]["models_to_exclude"], invert=True)
            self.realz_train[asset] = self.realz_train[asset][mask]
        
        # track sim iterations
        self.sim_iter = {}
        for asset in self.gen_sim_input:
            self.sim_iter[asset] = 0 # reset at len(self.realz_train)
        self.total_num_sim = 0
        
        # random number generator
        self.rng = np.random.default_rng(env_config.worker_index)
          
    def _setup_spaces(self):
        
        self.num_obs_data = (3 * self.num_prod + 2 * self.num_inj + 1) * self.sim_input["num_run_per_step"]
        if self.sim_input["epl_mode"] == "irr":
            self.num_obs_data += (2*self.sim_input["num_run_per_step"]) 
        
        if self.sim_input["reg_ctrl"]:
            self.action_space = spaces.Box(-1.0, +1.0, shape=[self.max_avail_act], dtype=np.float32)
            self.num_reg_obs_data = self.num_obs_data + (self.num_wells + 1) * self.sim_input["num_run_per_step"]
            #self.observation_space = spaces.Box(-10, 10, shape=(self.num_reg_obs_data,))
            self.observation_space = spaces.Dict({"states": spaces.Box(-10, 10, shape=(self.num_reg_obs_data,)), \
                                                 "well_ids": spaces.Box(0, self.num_wells, shape=(self.max_avail_act,)),\
                                                 "active_wells": spaces.Box(0, 1, shape=(self.max_avail_act,))})
        else:
            self.action_space = spaces.Box(0.0, +1.0, shape=[self.max_avail_act], dtype=np.float32)   
            self.observation_space = spaces.Box(-10, 10, shape=(self.num_obs_data,))
  
    def set_task(self):
        self.total_num_sim += self.sim_input["epl_start_iter"]
  
    def reset(self, realz=None, reg_limit=None, irr_min=None, asset=None):   
        
        # helps with memory issues
        #self.auto_garbage_collect()
        
        # select asset
        if asset == None:
            asset_ind = self.rng.choice(self.num_assets)
            self.asset = f"res_{asset_ind+1}"
        else:
            self.asset = asset
        self.sim_input = self.gen_sim_input[self.asset]
        
        # select realization  
        if realz == None:
            if self.sim_iter[self.asset] == self.realz_train[self.asset].shape[0]:
                self.sim_iter[self.asset] = 0
                self.rng.shuffle(self.realz_train[self.asset])
            
            self.sim_input["realz"] = int(self.realz_train[self.asset][self.sim_iter[self.asset]])
            self.sim_iter[self.asset] += 1
     
        else:
            self.sim_input["realz"] = realz
        
        # regularize well controls
        if self.sim_input["reg_ctrl"]:
            if reg_limit == None:
                self.reg_limit = self.rng.choice(self.sim_input["ctrl_regs"])
            else:
                self.reg_limit = reg_limit
        
        # irr
        if self.sim_input["epl_mode"] == "irr":
            if irr_min == None:
                self.irr_min = self.rng.choice(self.sim_input["irr_min_list"])
            else:
                self.irr_min = irr_min
          
        # reset
        self.res_sim.reset_vars(self.sim_input)
        self.hist.reset()
        self.cum_reward = 0
        self.cont_step = -1
        self.total_num_sim += 1
        self.prev_irr = 0
        
        # run historical period (assumes length of hist = length of ctrl step) #TODO: make general
        hist_action = np.pad(self.sim_input["hist_ctrl_scaled"], \
                             (0, self.max_avail_act-self.sim_input["hist_ctrl_scaled"].shape[0]))
        observation, _, _, _ = self.step(hist_action)

        return observation
    
    def step(self, action):
        assert self.action_space.contains(action)
   
        # select available actions
        action = action[:self.sim_input["num_wells"]]

        # advance to next control step
        self.cont_step += 1

        if not self.sim_input["reg_ctrl"] or self.cont_step == 0: #history/no regularization
            reg_action = action
        else:
            if self.cont_step == 1: #first opt control step for reg
                reg_action = (action + 1) / 2   # smoothen the control by mapping from [-1, 1] to [0,1]
            else:
                reg_action = self.hist.actions[-1] * (1 + action * self.reg_limit)
                reg_action = reg_action.clip(0, 1)

        
        # add action to history
        self.hist.actions.append(reg_action)
        
        # change well controls
        controls = reg_action * (self.sim_input["upper_bound"] - self.sim_input["lower_bound"]) + self.sim_input["lower_bound"]
        self.res_sim.set_well_control(controls)
       
        # run simulation
        end_of_lease = self.res_sim.run_single_ctrl_step()
        
        # calculate reward and irr
        reward, irr = self.res_sim.calculate_npv_and_irr()
        self.cum_reward += reward
        
        # check if end of project
        pre_check = (self.total_num_sim > self.sim_input["epl_start_iter"] and self.cont_step > 2)
        epl_terminate_neg_npv = (self.sim_input["epl_mode"] in ["negative_npv", "irr"] and pre_check and reward <= 0)
        epl_terminate_irr = (self.sim_input["epl_mode"] == "irr" and pre_check and self.prev_irr > irr and \
                             (irr - self.sim_input["irr_eps"]) < self.irr_min)                                  
        self.prev_irr = irr
        if end_of_lease or epl_terminate_neg_npv or epl_terminate_irr:
            done = True
        else:
            done = False
         
        # get observation
        observation = self.get_observation()  
             
        return observation, reward, done, {} 
    
    class History():
        def __init__(self):
            self.reset()

        def reset(self):
            self.scaled_state = []
            self.unscaled_state = []
            self.reward_dollar = []
            self.actions = []
            self.done = []
            
    def experiment(self, actions):
        # advance from "current state" according to actions
        self.hist.reset()
        
        unscaled_obs = self.get_unscaled_state()
        self.hist.unscaled_state.append(unscaled_obs)
        obs = self.get_observation()
        self.hist.scaled_state.append(obs)

        for i_action in actions:
            obs, reward, done, _ = self.step(i_action)
            self.hist.scaled_state.append(obs)
            self.hist.reward_dollar.append(reward)
            self.hist.done.append(done)
            
            unscaled_obs = self.get_unscaled_state()
            self.hist.unscaled_state.append(unscaled_obs)
    
    def get_unscaled_state(self):
        _ , unscaled_state = self.res_sim.get_observation()
        
        return unscaled_state
                        
    def get_observation(self): 
        sim_obs, _ = self.res_sim.get_observation()
        
        # asset obs
        obs = np.array([])
        for asset in self.gen_sim_input:        
            if asset != self.asset: # inactive asset
                asset_obs = self.asset_obs[asset]
            else: # active asset 
                asset_obs = sim_obs.reshape(self.sim_input["num_run_per_step"], -1)
                if self.sim_input["reg_ctrl"]:
                    reg_prev_act = np.array([self.hist.actions[-1]]*self.sim_input["num_run_per_step"])
                    asset_obs = np.hstack([asset_obs, reg_prev_act])
            
            # aggregate
            if obs.shape[0] == 0:
                obs = asset_obs
            else:
                obs = np.hstack([obs, asset_obs])
    
        # add time
        norm_time = self.cont_step / (self.sim_input["num_cont_step"] - 1)
        time_obs = np.expand_dims(np.array([norm_time]*self.sim_input["num_run_per_step"]), axis=1)
        obs = np.hstack([obs, time_obs])
        
        # add reg limit
        if self.sim_input["reg_ctrl"]:
            norm_reg = self.reg_limit/self.sim_input["ctrl_regs"][-1]
            reg_obs = np.expand_dims(np.array([norm_reg]*self.sim_input["num_run_per_step"]), axis=1)
            obs = np.hstack([obs, reg_obs])
        
        # add irr
        if self.sim_input["epl_mode"] == "irr":
            low_irr, high_irr = self.sim_input["irr_min_list"][0], self.sim_input["irr_min_list"][-1]
            norm_irr_min = (self.irr_min - low_irr) / (high_irr - low_irr)
            irr_obs = np.concatenate(([self.prev_irr], [norm_irr_min]))
            irr_obs = np.array([irr_obs]*self.sim_input["num_run_per_step"])
            obs = np.hstack([obs, irr_obs])
        
        return {"states": obs.flatten().astype('float32'), "well_ids": self.well_ids[self.asset].astype('float32'),\
                "active_wells": self.well_ids[self.asset].astype('bool').astype('float32')}
    
    
    def auto_garbage_collect(self, pct=55.0):
        """
        auto_garbage_collection - Call the garbage collection if memory used is greater than 65% of total available memory.
                                  This is called to deal with an issue in Ray not freeing up used memory.

            pct - Default value of 65%.  Amount of memory in use that triggers the garbage collection call.
        """
        if psutil.virtual_memory().percent >= pct:
            gc.collect()     
