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
from elicitation import parameter_elicitation_utilities_option1, parameter_elicitation_utilities_tanh
from elicit_lambda import elicit_lambda

import logging
import datetime 
import os

import yaml

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)



def update_influence_diagram(model_type = None, value_function = None, elicit = None, ref_patient_chars = None, new_test = None, logger = None):

    logger.info(f"Model type: {model_type}")

    # Read the network -----------------------------------------------------
    logger.info("Reading network...")
    net = pysmile.Network()
    net.read_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    logger.info("Calculating relative pointwise conditional mutual information values...")

    try:
        net.delete_arc("Results_of_Screening", "Colonoscopy")
    except:
        logger.info("No arc to delete")

    value_function = "rel_pcmi"
    df_value_scr, df_value_col = save_info_values(net, value_function = value_function, weighted=False)
    net2 = info_value_to_net(df_value_scr, df_value_col, net)



    # ----------------------------------------------------------------------
    if model_type == "tanh":
        logger.info("Defining value of comfort...")
        rho_4 = 0.6
        rho_3 = 0.55
        rho_2 = 0.50
        rho_1 = 0.40
        arr_comft = np.array([rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4,
                            rho_4, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_2, rho_1, rho_2, rho_1, # Added discomfort when colonoscopy is not mandatory
                            rho_4, rho_4, rho_3, rho_3, rho_3, rho_3, rho_3, rho_3, rho_3, rho_3, rho_2, rho_2, rho_2, rho_2,]) #No added discomfort when colonoscopy is mandatory
        net2.set_node_definition("Value_of_comfort", arr_comft)

        net2.set_mau_expressions(node_id = "V", expressions = [f"((8131.71-COST)/8131.71)*Tanh(VALUE*Value_of_comfort)"])

    elif model_type == "linear":
        logger.info("Eliciting value of comfort...")

        if elicit == True:

            lambdas = elicit_lambda(patient_chars = ref_patient_chars,
                                    net = net2, logging = logger)
            
            net.set_node_definition("Value_of_comfort", lambdas)
            net.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*VALUE - Log10(COST+1)"])

        else:
            try:
                lambdas = net2.get_node_value("Value_of_comfort")

                logger.info("No elicitation of lambda values, taking default values...")
            
            except:
                logger.info("No default values found, setting custom values...")
                rho_4 = 8
                rho_3 = 6.5
                rho_2 = 6.25
                rho_1 = 6 # Not a bad choice of values. Try to justify them correctly.
                arr_comft = np.array([rho_4, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_2, rho_1, rho_2, rho_1,])
                net2.set_node_definition("Value_of_comfort", arr_comft)

                net2.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*VALUE - Log10(COST+1)"])

                lambdas = net2.get_node_value("Value_of_comfort")



    # ----------------------------------------------------------------------
    if new_test:
        net.add_outcome("Screening", "New_test")
        logger.info("Adding new test values...")
        net2 = values_for_new_test(net2, config = cfg)
        df_value_scr, df_value_col = save_info_values(net, value_function = value_function, new_test=True, weighted=False)
        net2 = info_value_to_net(df_value_scr, df_value_col, net2)



    # ----------------------------------------------------------------------
    logger.info("Saving network...")
    if new_test:
        net2.write_file(f"decision_models/DM_screening_{value_function}_{model_type}_new_test.xdsl")
    else:
        net2.write_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    logger.info("Plotting info functions...")
    if new_test:
        plot_cond_mut_info(net2, subtitle='new_test')
        plot_relative_cond_mut_info(net2, subtitle = 'new_test', zoom = (0.0001, 0.1))
    else: 
        plot_cond_mut_info(net2, subtitle='')
        plot_relative_cond_mut_info(net2, subtitle = '', zoom = (0.0001, 0.1))
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    logger.info("Calculating final utilities...")

    if model_type == "tanh":
        params = parameter_elicitation_utilities_tanh(PE_info = cfg["PE_info"], PE_cost = cfg["PE_cost"], rho_comfort = lambdas[2])
    elif model_type == "linear":
        params = parameter_elicitation_utilities_option1(PE_info = cfg["PE_info"], PE_cost = cfg["PE_cost"], rho_comfort = lambdas[2], logging = logger)

    if params is None:
        logger.warning("Please try another initial value for the system of equations...")
        exit()

    else:
        logger.info(f"Parameters found: {params}")
        net2.set_mau_expressions(node_id = "U", expressions = [f"({params[0]} - {params[1]}*Exp( - {params[2]} * V))"])
        
        if new_test:
            net2.write_file(f"decision_models/DM_screening_{value_function}_{model_type}_new_test.xdsl")
        else:
            net2.write_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")
        logger.info("Done!")
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    logger.info("Calculating utilities for patient X...")

    net.clear_all_evidence()

    for key, value in ref_patient_chars.items():
        net.set_evidence(key, value)

    net.update_beliefs()

    # pdb.set_trace()
    if len(net2.get_node_value("U")) == 14:
        vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC"]
        vars2 = ["No colonoscopy", "Colonoscopy"]

        comb = list(itertools.product(vars1, vars2))

        index = pd.MultiIndex.from_tuples(comb)
        arr = np.array(net2.get_node_value("U"))

        df_U = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)

    elif len(net2.get_node_value("U")) == 16:
        vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC", "New_test"]
        vars2 = ["No colonoscopy", "Colonoscopy"]

        comb = list(itertools.product(vars1, vars2))

        index = pd.MultiIndex.from_tuples(comb)
        arr = np.array(net2.get_node_value("U"))

        df_U = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)


    if new_test:
        df_U.to_csv("U_values_new_test.csv")
    else:  
        df_U.to_csv("U_values.csv")

    logger.info(f"\n {df_U}")
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    net2.add_arc("Results_of_Screening", "Colonoscopy")
    net2.update_beliefs()

    if new_test:
        vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC", "New_test"]
    else:
        vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC"]

    vars2 = ["No pred", "Pred False", "Pred True"]
    vars3 = ["No colonoscopy", "Colonoscopy"]

    comb = list(itertools.product(vars1, vars2, vars3))

    index = pd.MultiIndex.from_tuples(comb)
    arr = np.array(net2.get_node_value("U"))

    df_U_ext = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)
    logger.info(f"\n {df_U_ext}")

    if new_test:
        df_U_ext.to_csv("U_values_cond_new_test.csv")
    else:
        df_U_ext.to_csv("U_values_cond.csv")
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    logger.info("Done!")
    # ----------------------------------------------------------------------

    return net2



def values_for_new_test(net, config):
    num_scr_tests = len(net.get_outcome_ids("Screening"))


    # ---Set comfort ----
    comfort_definition = net.get_node_definition("Value_of_comfort")

    value_of_comfort_new_test = comfort_definition[2]
    comfort_definition[-2] = value_of_comfort_new_test
    comfort_definition[-1] = comfort_definition[1]

    net.set_node_definition("Value_of_comfort", comfort_definition)


    # --- Set cost ---.
    cost_definition = net.get_node_definition("Cost_of_Screening")

    cost_new_test = config["cost_new_test"]
    cost_definition[-2] = cost_new_test
    cost_definition[-1] = cost_definition[1] + cost_new_test

    net.set_node_definition("Cost_of_Screening", cost_definition)


    # --- Set sensitivity and specificity ---

    sens_spec_arr = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,-1,3)

    sens_spec_arr[0,-1,:] = [0, config["specificity_new_test"], 1 - config["specificity_new_test"]]
    sens_spec_arr[1,-1,:] = [0, 1 - config["sensitivity_new_test"], config["sensitivity_new_test"]]

    net.set_node_definition("Results_of_Screening", sens_spec_arr.reshape(-1))


    # ---- Set complications ----
    complications_arr = np.array(net.get_node_definition("Complications")).reshape(num_scr_tests, 2, -1)

    complications_arr[-1, 0] = np.array([1,0,0,0,0])
    complications_arr[-1, 1] = np.array(net.get_node_definition("Complications")).reshape(num_scr_tests, 2, -1)[0,1]

    net.set_node_definition("Complications", complications_arr.reshape(-1))

    return net