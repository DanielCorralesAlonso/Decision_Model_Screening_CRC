import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools
from plots import plot_cond_mut_info, plot_relative_cond_mut_info
from save_info_values import save_info_values
np.seterr(divide='ignore', invalid = 'ignore')

from network_functions import calculate_network_utilities, new_screening_strategy, old_screening_strategy, create_folders_logger
from simulations import plot_classification_results
from plots import plot_estimations_w_error_bars, plot_screening_counts
from preprocessing import preprocessing
from update import update_influence_diagram

import logging
import datetime 
import os

import pdb

import yaml

import subprocess

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)


def sens_analysis_elicitation(
        net = None,
        operational_limit = cfg["operational_limit"],
        operational_limit_comp = cfg["operational_limit_comp"], 
        single_run = cfg["single_run"],
        num_runs = cfg["num_runs"],
        use_case_new_test = cfg["new_test"],
        all_variables = cfg["all_variables"],
        from_elicitation = cfg["from_elicitation"], 
        lambda_list = cfg["lambda_list"],
        logger = None,
        log_dir = None,
        run_label = '',
        label = '',
        output_dir = None
    ):

    if "inf" in operational_limit.values():
        operational_limit = {k: np.inf if v == "inf" else v for k, v in operational_limit.items()}
    if "inf" in operational_limit_comp.values():
        operational_limit_comp = {k: np.inf if v == "inf" else v for k, v in operational_limit_comp.items()}


    if logger == None:
        logger, log_dir = create_folders_logger(single_run=single_run, label="sens_analysis_elicitation_", output_dir = output_dir )
    else:
        log_dir = os.path.join(log_dir, run_label)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logger.info("Configuration variables of interest:")
    logger.info(f"Single run: {single_run}")
    logger.info(f"Use all variables: {all_variables}")
    logger.info(f"Use case with new test: {use_case_new_test}")
    logger.info(f"PE method: {cfg['rel_point_cond_mut_info']}")
    logger.info(f"Change lambdas? {cfg['noise']}")
    logger.info(f"Read lambda list from config? {cfg['lambda_list_from_config']}")
    logger.info(f"Lambdas: {lambda_list}")

    output_dir = os.path.join(output_dir, '_run')

    update_influence_diagram(
        model_type = cfg["model_type"],
        value_function = cfg["value_function"],
        elicit = cfg["elicit"],
        noise = cfg["noise"],
        calculate_info_values= cfg["calculate_info_values"],
        ref_patient_chars = cfg["patient_chars"],
        predefined_lambdas = lambda_list,
        new_test = cfg["new_test"],
        logger = logger,
        output_dir = output_dir
    )


    logger.info("Reading the network file...")
    if net == None:
        net = pysmile.Network()
        if use_case_new_test == True:
            file_location = f"{output_dir}/decision_models/DM_screening_rel_point_cond_mut_info_linear_new_test.xdsl"
        elif from_elicitation == True:
            file_location = f"{output_dir}/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl"
        else:
            file_location = f"{output_dir}/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl"
        net.read_file(file_location)
        logger.info(f"Located at: {file_location}")

    lambdas_comfort = net.get_node_definition("Value_of_comfort")
    logger.info(f"Comfort values: 1 - {lambdas_comfort[1]}, 2 - {lambdas_comfort[-4]}, 3 - {lambdas_comfort[2]}, 4 - {lambdas_comfort[0]}")

    df_test = pd.read_csv("private/df_2016.csv")
    df_test = preprocessing(df_test)
    df_test = df_test.rename(columns = {"Hyperchol.": "Hyperchol_"})

    # Just keep variables that influence the decision
    if all_variables == False:
        df_test.drop(columns = ["Hyperchol_", "Hypertension", "Diabetes", "SES", "Anxiety", "Depression"], inplace = True)
        logger.info("Only variables that influence the decision are kept in the dataframe for calculation of utilities.")
    else:
        logger.info(
            "All variables are kept in the dataframe for calculation of utilities.")
        pass


    df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test)
    lamdbas = net.get_node_definition("Value_of_comfort")
    lambda_list = [lamdbas[1], lamdbas[-4], lamdbas[2], lamdbas[0]]
    
    if use_case_new_test == True:
        operational_limit["New_test"] = 20000

    plot_screening_counts(counts, possible_outcomes, operational_limit.values(), log_dir=log_dir, lambda_list = lambda_list)

    print("Counts:" , counts)

    return




if __name__ == "__main__":
    sens_analysis_elicitation()


