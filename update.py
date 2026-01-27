import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools
from plots import plot_cond_mut_info, plot_relative_cond_mut_info
from save_info_values import save_info_values
np.seterr(divide='ignore', invalid = 'ignore')

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures
from functions import system_of_eq, tanh_fun
from elicitation import parameter_elicitation_utilities_linear, parameter_elicitation_utilities_tanh
from elicit_lambda import elicit_lambda

import logging
import datetime 
import os

import yaml

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)



def update_influence_diagram(model_type = None, value_function = None, elicit = None, noise=None, calculate_info_values = None ,
                              ref_patient_chars = None, new_test = None, sens_analysis_metrics = None, logger = None, output_dir = None,
                              change_risk_param = False, rho_param = None, exogenous_var_prob = None, predefined_lambdas = None):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}/decision_models"):
        os.makedirs(f"{output_dir}/decision_models")
    if not os.path.exists(f"{output_dir}/output_images"):
        os.makedirs(f"{output_dir}/output_images")
    if not os.path.exists(f"{output_dir}/output_data"):
        os.makedirs(f"{output_dir}/output_data")


    logger.info(f"Model type: {model_type}")

    # Read the network -----------------------------------------------------
    logger.info("Reading network...")
    net = pysmile.Network()
    net.read_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")
    # ----------------------------------------------------------------------


    if exogenous_var_prob is not None:
        new_cpt = net.get_node_definition('CRC')
        new_cpt[0] = 1 - exogenous_var_prob
        new_cpt[1] = exogenous_var_prob
        net.set_node_definition('CRC', new_cpt)

    # ----------------------------------------------------------------------
    if sens_analysis_metrics == "lower":
        net.set_node_definition("Results_of_Screening", cfg["sens_analysis_metrics_lower"]["screening"])
        net.set_node_definition("Results_of_Colonoscopy", cfg["sens_analysis_metrics_lower"]["colonoscopy"])

    if sens_analysis_metrics == "upper":
        net.set_node_definition("Results_of_Screening", cfg["sens_analysis_metrics_upper"]["screening"])
        net.set_node_definition("Results_of_Colonoscopy", cfg["sens_analysis_metrics_upper"]["colonoscopy"])
    

    # ----------------------------------------------------------------------
    if calculate_info_values:

        logger.info("Calculating information values...")

        df_value = save_info_values(net, value_function = value_function, output_dir=output_dir)


        net2 = info_value_to_net( df_value, net)
        df_value.to_csv(f"{output_dir}/output_data/INFO_node.csv")

    else:
        net2 = net
        net2.update_beliefs()

    # ----------------------------------------------------------------------
    
    if predefined_lambdas is None:
        if model_type == "linear":
            logger.info("Eliciting value of comfort...")

            if elicit == True:

                lambdas = elicit_lambda(patient_chars = ref_patient_chars, value_function = value_function,
                                        net = net2, logging = logger)
                
                
                
                net.set_node_definition("Value_of_comfort", lambdas)
                net.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*INFO - Log10(COST+1)"])

            else:
                try:
                    lambdas = net2.get_node_value("Value_of_comfort")
                    logger.info("No elicitation of lambda values, taking default values...")
                    
                        
                    if cfg["lambda_list_from_config"] == True:
                        lambda_list = cfg["lambda_list"]
                    else:
                        lambda_list = [lambdas[1], lambdas[-2], lambdas[2]]
                        lambda_list_mod = np.random.normal(lambda_list, cfg['noise_std'])
                        while not np.array_equal(np.sort(lambda_list_mod), lambda_list_mod):
                            lambda_list_mod = np.random.normal(lambda_list, cfg['noise_std'])

                        lambda_list = lambda_list_mod
                    
                    lambdas = np.array([np.ceil(lambda_list[2]), lambda_list[0], lambda_list[2], lambda_list[0], 
                        lambda_list[2], lambda_list[0], lambda_list[2], lambda_list[0], 
                        lambda_list[2], lambda_list[0], lambda_list[1], lambda_list[0], lambda_list[1], lambda_list[0],])

                    net2.set_node_definition("Value_of_comfort", lambdas)
                    net2.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*INFO - Log10(COST+1)"])
                
                    logger.info(f"Lambda values: {lambdas}")

                except:
                    logger.info("No default values found, setting custom values...")
                    lambda_list = cfg["lambda_list"]
                    lambdas = np.array([np.ceil(lambda_list[2]), lambda_list[0], lambda_list[2], lambda_list[0], 
                            lambda_list[2], lambda_list[0], lambda_list[2], lambda_list[0], 
                            lambda_list[2], lambda_list[0], lambda_list[1], lambda_list[0], lambda_list[1], lambda_list[0],])

                    net2.set_node_definition("Value_of_comfort", lambdas)

                    net2.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*INFO - Log10(COST+1)"])

                    lambdas = net2.get_node_value("Value_of_comfort")

    else:
        lambda_list = predefined_lambdas
        logger.info(f"Lambdas: {lambda_list}")
        lambdas = np.array([np.ceil(lambda_list[2]), lambda_list[0], lambda_list[2], lambda_list[0], 
                            lambda_list[2], lambda_list[0], lambda_list[2], lambda_list[0], 
                            lambda_list[2], lambda_list[0], lambda_list[1], lambda_list[0], lambda_list[1], lambda_list[0],])

        net.set_node_definition("Value_of_comfort", lambdas)
        net.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*INFO - Log10(COST+1)"])




    # ----------------------------------------------------------------------
    
    if new_test:
        new_tests_config = cfg.get("new_tests", {})
        if new_tests_config:
            new_test = True # Ensure downstream logic treats this as having new tests
            for test_name in new_tests_config:
                if test_name not in net.get_outcome_ids("Screening"):
                    net.add_outcome("Screening", test_name)

            logger.info(f"Adding values for {len(new_tests_config)} new tests...")
            net2 = values_for_new_tests(net2, new_tests_config = new_tests_config)
            df_value = save_info_values(net2, value_function = value_function, new_test=True, output_dir = output_dir)
            net2 = info_value_to_net(df_value, net2)



    # ----------------------------------------------------------------------
    logger.info("Saving network...")
    if new_test:
        net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}_new_test.xdsl")

    if sens_analysis_metrics == "lower":
        net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}_sens_analysis_lower.xdsl")
    elif sens_analysis_metrics == "upper":
        net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}_sens_analysis_upper.xdsl")
    elif not new_test:
        net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}.xdsl")
    # ----------------------------------------------------------------------

    
    # ----------------------------------------------------------------------
    logger.info("Plotting info functions...")
    if new_test:
        # plot_cond_mut_info(net2, subtitle='new_test', output_dir = output_dir)
        plot_relative_cond_mut_info(net2, subtitle = 'new_test', zoom = (0.001, 0.1), step = 0.001, output_dir = output_dir)
    if sens_analysis_metrics == "lower":
        # plot_cond_mut_info(net2, subtitle='sens_analysis_lower', output_dir = output_dir)
        plot_relative_cond_mut_info(net2, subtitle = 'sens_analysis_lower', zoom = (0.001, 0.1), step = 0.001, output_dir = output_dir)
    if sens_analysis_metrics == "upper":
        # plot_cond_mut_info(net2, subtitle='sens_analysis_upper', output_dir = output_dir)
        plot_relative_cond_mut_info(net2, subtitle = 'sens_analysis_upper', zoom = (0.001, 0.1), step = 0.001, output_dir = output_dir)
    else: 
        # plot_cond_mut_info(net2, subtitle='', output_dir = output_dir)
        plot_relative_cond_mut_info(net2, subtitle = '', zoom = (0.001, 0.1), step = 0.001, output_dir = output_dir)
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    logger.info("Calculating final utilities...")
    

    if model_type == "linear":
        params = parameter_elicitation_utilities_linear(net2, PE = cfg[value_function]["PE_prob"], PE_info = cfg[value_function]["PE_info"], PE_cost = cfg[value_function]["PE_cost"], rho_comfort = lambdas[2], value_function = value_function, logging = logger)

    if params is None:
        logger.warning("Please try another initial value for the system of equations...")
        exit()

    else:
        logger.info(f"Parameters found: {params}")

        if change_risk_param:
            params = params[:2] + [rho_param]

        net2.set_mau_expressions(node_id = "U", expressions = [f"{params[0]} - {params[1]}*Exp( - {params[2]} * V)"])

        if new_test:
            net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}_new_test.xdsl")

        if sens_analysis_metrics == "lower":
            net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}_sens_analysis_lower.xdsl")
        elif sens_analysis_metrics == "upper":
            net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}_sens_analysis_upper.xdsl")
        elif not new_test:
            net2.write_file(f"{output_dir}/decision_models/DM_screening_{value_function}_{model_type}.xdsl")
        logger.info("Done!")
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    logger.info("Calculating utilities for patient X...")

    net2.clear_all_evidence()

    for key, value in ref_patient_chars.items():
        net2.set_evidence(key, value)

    net.update_beliefs()


    vars1 = net2.get_outcome_ids("Screening")
    vars2 = net2.get_outcome_ids("Results_of_Screening")
    vars3 = net2.get_outcome_ids("Colonoscopy")

    comb = list(itertools.product(vars1, vars2, vars3))

    index = pd.MultiIndex.from_tuples(comb)
    arr = np.array(net2.get_node_value("U"))

    df_U = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)


    if new_test:
        df_U.to_csv(f"{output_dir}/output_data/U_values_new_test.csv")
    if sens_analysis_metrics == "lower":
        df_U.to_csv(f"{output_dir}/output_data/U_values_sens_analysis_lower.csv")
    if sens_analysis_metrics == "upper":
        df_U.to_csv(f"{output_dir}/output_data/U_values_sens_analysis_upper.csv")
    else:  
        df_U.to_csv(f"{output_dir}/output_data/U_values.csv")

    logger.info(f"\n {df_U}")
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    logger.info("Calculating best screening strategy for patient X...")

    net2.clear_all_evidence()

    for key, value in ref_patient_chars.items():
        net2.set_evidence(key, value)

    net2.update_beliefs()


    vars1 = net2.get_outcome_ids("Screening")

    arr = np.array(net2.get_node_value("Screening"))

    df_scr = pd.DataFrame(arr.reshape(1,-1), index=["Screening"], columns=vars1)


    if new_test:
        df_scr.to_csv(f"{output_dir}/output_data/Screening_util_new_test.csv")
    if sens_analysis_metrics == "lower":
        df_scr.to_csv(f"{output_dir}/output_data/Screening_util_sens_analysis_lower.csv")
    if sens_analysis_metrics == "upper":
        df_scr.to_csv(f"{output_dir}/output_data/Screening_util_sens_analysis_upper.csv")
    else:  
        df_scr.to_csv(f"{output_dir}/output_data/Screening_util.csv")

    logger.info(f"\n {df_scr}")

    
    for handler in logger.handlers:
        handler.close()          # Close the handler
        logger.removeHandler(handler)  # Remove the handler from the logger

    return net2



def values_for_new_tests(net, new_tests_config):
    screening_outcomes = net.get_outcome_ids("Screening")
    num_scr_tests = len(screening_outcomes)

    # --- Definitions ---
    comfort_definition = np.array(net.get_node_definition("Value_of_comfort"))
    cost_definition = np.array(net.get_node_definition("Cost_of_Screening"))
    
    sens_spec_arr = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,-1,3)
    complications_arr = np.array(net.get_node_definition("Complications")).reshape(num_scr_tests, 2, -1)

    for test_name, params in new_tests_config.items():
        if test_name not in screening_outcomes:
            continue
        
        idx = screening_outcomes.index(test_name)
        ref_idx = params.get("comfort_level", 3) # Default to gFOBT-like comfort if not specified

        # --- Set comfort ---
        comfort_definition[2 * idx] = cfg["lambda_list"][ref_idx]
        comfort_definition[2 * idx + 1] = cfg["lambda_list"][1]

        # --- Set cost ---
        cost_val = params.get("cost", 0)
        cost_definition[2 * idx] = cost_val
        cost_definition[2 * idx + 1] = cost_definition[1] + cost_val

        # --- Set sensitivity and specificity ---
        sens = params.get("sensitivity", 0)
        spec = params.get("specificity", 0)
        
        sens_spec_arr[0, idx, :] = [0, spec, 1 - spec]
        sens_spec_arr[1, idx, :] = [0, 1 - sens, sens]

        # --- Set complications ---
        complications_arr[idx, 0] = np.array([1,0,0,0,0])
        complications_arr[idx, 1] = complications_arr[0, 1]

    net.set_node_definition("Value_of_comfort", comfort_definition)
    net.set_node_definition("Cost_of_Screening", cost_definition)
    net.set_node_definition("Results_of_Screening", sens_spec_arr.reshape(-1))
    net.set_node_definition("Complications", complications_arr.reshape(-1))

    return net