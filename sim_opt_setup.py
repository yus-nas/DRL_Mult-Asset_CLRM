import numpy as np


def Sim_opt_setup(res):
    sim_input = {}
    
    # Grid
    sim_input["nx"] =  60 
    sim_input["ny"] =  60
    sim_input["nz"] =  1
    sim_input["dx"] =  60
    sim_input["dy"] =  60
    sim_input["dz"] =  12

    # Initialization
    sim_input["datum"] = 3000
    sim_input["Pi"] = 350
    sim_input["Swi"] =  0.15
    
    # physics file
    sim_input["physics_file"] = "physics.in"

    # Rock
    sim_input["depth"] = 3000
    sim_input["poro"] = 0.2 
    sim_input["kz"] = 100 
    sim_input["actnum"] = np.ones(sim_input["nx"]*sim_input["ny"])
    
    # realization
    sim_input["realz"] = 2
    sim_input["realz_path"] = f"/scratch/users/nyusuf/Research_projects/Models/Multi_asset_2D/{res}/Ensemble/_multperm"
    sim_input["cluster_labels"] = np.loadtxt(f"Model_clusters/cluster_label_{res}_2d.txt").astype(int)
    sim_input["models_to_exclude"] = np.loadtxt(f"Model_clusters/rep_models_{res}_2d.txt").astype(int)


    # well
    sim_input["well_radius"] = 0.3048*0.5
    sim_input["prod_bhp"] = 345
    sim_input["inj_bhp"] = 400
    sim_input["skin"] = 0

    # timing
    sim_input["total_time"] = 4000
    sim_input["num_run_per_step"] = 5
    sim_input["runtime"] = 40
    sim_input["len_cont_step"] = sim_input["num_run_per_step"] * sim_input["runtime"]
    sim_input["num_cont_step"] = sim_input["total_time"] / sim_input["len_cont_step"]

    # economics
    sim_input["oil_price"] = 70
    sim_input["capex"] = 900e6 
    sim_input["opex"] = 15e6 / 365 # $/day
    sim_input["discount_rate"] = 0.1
    sim_input["wat_prod_cost"] = 7
    sim_input["wat_inj_cost"] = 7
    sim_input["npv_scale"] = 1e9

    # constraint
    sim_input["max_liq_prod_rate"] = 1526
    sim_input["max_water_inj_rate"] = 1526
    sim_input["water_cut_limit"] = 0.98
    sim_input["reg_ctrl"] = False
    sim_input["ctrl_regs"] = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.27]
    
    # economic project life 
    sim_input["epl_mode"] = None  # None (end of lease) or negative_npv or irr
    sim_input["epl_start_iter"] = 0
    sim_input["irr_min_list"] = [0.12, 0.16, 0.20, 0.24]
    sim_input["irr_eps"] = 0.0025
    
    # noise
    sim_input["noise"] = True
    sim_input["std_rate_min"] = 1.5
    sim_input["std_rate_max"] = 7.94936
    sim_input["std_rate"] = 0.05
    sim_input["std_pres"] = 0.344738

    # solver
    sim_input["first_ts"] = 0.01
    sim_input["mult_ts"] = 2
    sim_input["max_ts"] = 40        # days  # max time step 
    sim_input["tolerance_newton"] = 1e-6
    sim_input["tolerance_linear"] = 1e-8

    # well completion
    
    sim_input["well_comp"] = []
    
    if res == "res_1":
        sim_input["well_comp"].append(['P1', 'PROD', 11, 11, 1, 1])
        sim_input["well_comp"].append(['P2', 'PROD', 9, 53, 1, 1])
        sim_input["well_comp"].append(['P3', 'PROD', 31, 31, 1, 1])
        sim_input["well_comp"].append(['P4', 'PROD', 51, 11, 1, 1])
        sim_input["well_comp"].append(['P5', 'PROD', 51, 51, 1, 1])
        sim_input["well_comp"].append(['I1', 'INJ', 8, 24, 1, 1])
        sim_input["well_comp"].append(['I2', 'INJ', 36, 11, 1, 1])
        sim_input["well_comp"].append(['I3', 'INJ', 19, 41, 1, 1])
        sim_input["well_comp"].append(['I4', 'INJ', 51, 37, 1, 1])
    
    elif res == "res_2":
        sim_input["well_comp"].append(['P1', 'PROD', 13, 7, 1, 1])
        sim_input["well_comp"].append(['P2', 'PROD', 13, 21, 1, 1])
        sim_input["well_comp"].append(['P3', 'PROD', 13, 33, 1, 1])
        sim_input["well_comp"].append(['P4', 'PROD', 13, 51, 1, 1])
        sim_input["well_comp"].append(['P5', 'PROD', 25, 7, 1, 1])
        sim_input["well_comp"].append(['P6', 'PROD', 25, 36, 1, 1])
        sim_input["well_comp"].append(['P7', 'PROD', 41, 7, 1, 1])
        sim_input["well_comp"].append(['P8', 'PROD', 41, 21, 1, 1])
        sim_input["well_comp"].append(['P9', 'PROD', 41, 36, 1, 1])
        sim_input["well_comp"].append(['P10', 'PROD', 41, 53, 1, 1])
        sim_input["well_comp"].append(['P11', 'PROD', 53, 7, 1, 1])
        sim_input["well_comp"].append(['P12', 'PROD', 53, 46, 1, 1])
        sim_input["well_comp"].append(['I1', 'INJ', 25, 21, 1, 1])
        sim_input["well_comp"].append(['I2', 'INJ', 25, 53, 1, 1])
        sim_input["well_comp"].append(['I3', 'INJ', 53, 26, 1, 1])
        sim_input["well_comp"].append(['I4', 'INJ', 53, 57, 1, 1])
    
    elif res == "res_3":
        sim_input["well_comp"].append(['P1', 'PROD', 12, 50, 1, 1])
        sim_input["well_comp"].append(['P2', 'PROD', 24, 50, 1, 1])
        sim_input["well_comp"].append(['P3', 'PROD', 36, 50, 1, 1])
        sim_input["well_comp"].append(['P4', 'PROD', 48, 50, 1, 1])
        sim_input["well_comp"].append(['P5', 'PROD', 12, 10, 1, 1])
        sim_input["well_comp"].append(['P6', 'PROD', 24, 10, 1, 1])
        sim_input["well_comp"].append(['P7', 'PROD', 36, 10, 1, 1])
        sim_input["well_comp"].append(['P8', 'PROD', 48, 10, 1, 1])
        sim_input["well_comp"].append(['I1', 'INJ', 12, 30, 1, 1])
        sim_input["well_comp"].append(['I2', 'INJ', 30, 30, 1, 1])
        sim_input["well_comp"].append(['I3', 'INJ', 48, 30, 1, 1])
    
    elif res == "res_4":
        sim_input["well_comp"].append(['P1', 'PROD', 5, 55, 1, 1])
        sim_input["well_comp"].append(['P2', 'PROD', 39, 55, 1, 1])
        sim_input["well_comp"].append(['P3', 'PROD', 21, 39, 1, 1])
        sim_input["well_comp"].append(['P4', 'PROD', 55, 45, 1, 1])
        sim_input["well_comp"].append(['P5', 'PROD', 5, 21, 1, 1])
        sim_input["well_comp"].append(['P6', 'PROD', 39, 21, 1, 1])
        sim_input["well_comp"].append(['P7', 'PROD', 5, 5, 1, 1])
        sim_input["well_comp"].append(['P8', 'PROD', 21, 5, 1, 1])
        sim_input["well_comp"].append(['P9', 'PROD', 39, 5, 1, 1])
        sim_input["well_comp"].append(['I1', 'INJ', 21, 55, 1, 1])
        sim_input["well_comp"].append(['I2', 'INJ', 5, 39, 1, 1])
        sim_input["well_comp"].append(['I3', 'INJ', 39, 39, 1, 1])
        sim_input["well_comp"].append(['I4', 'INJ', 21, 21, 1, 1])
        sim_input["well_comp"].append(['I5', 'INJ', 55, 15, 1, 1])
    
    
    # Opt controls
    prod_bhp_bound = [280, 345]
    inj_bhp_bound = [355, 450]
    hist_control_prod = sim_input["prod_bhp"]
    hist_control_inj = sim_input["inj_bhp"]
    
    # num wells
    num_prod = 0
    num_inj = 0
    for comp in sim_input["well_comp"]:
        if comp[1] == 'PROD':
            num_prod += 1
        else:
            num_inj += 1
    
    sim_input["num_prod"] = num_prod
    sim_input["num_inj"] = num_inj
    sim_input["num_wells"] = num_prod + num_inj
    
    # Preprocess control info
    sim_input["lower_bound"] = np.concatenate(([prod_bhp_bound[0]]*num_prod, [inj_bhp_bound[0]]*num_inj))
    sim_input["upper_bound"] = np.concatenate(([prod_bhp_bound[1]]*num_prod, [inj_bhp_bound[1]]*num_inj))
    sim_input["hist_ctrl"] = np.concatenate(([hist_control_prod]*num_prod, [hist_control_inj]*num_inj))
    sim_input["hist_ctrl_scaled"] = (sim_input["hist_ctrl"]-sim_input["lower_bound"])/(sim_input["upper_bound"]-sim_input["lower_bound"])
    sim_input["hist_ctrl_scaled"] = sim_input["hist_ctrl_scaled"].astype('float32')

    # scaling factors
    sim_input["scaler"] = np.loadtxt(f"scaling_factors_{res}_2d.txt", delimiter=",")
    
    return sim_input