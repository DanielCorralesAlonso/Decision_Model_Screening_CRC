import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from network_functions import calculate_network_utilities, new_screening_strategy, old_screening_strategy, create_folders_logger
from simulations import plot_classification_results
from plots import plot_estimations_w_error_bars, plot_screening_counts

from preprocessing import preprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

import argparse
import pdb
import matplotlib
matplotlib.use('Agg')

import logging
import datetime 
import os

np.seterr(divide='ignore', invalid = 'ignore')


def use_case_new_strategy(net = None,
        file_location = None,
        operational_limit = cfg["operational_limit"],
        operational_limit_comp = cfg["operational_limit_comp"], 
        single_run = cfg["single_run"],
        num_runs = cfg["num_runs"],
        use_case_new_test = cfg["use_case_new_test"],
        all_variables = cfg["all_variables"],
        from_elicitation = cfg["from_elicitation"],  
        logger = None,
        log_dir = None,
        run_label = '',
        best_f1_score = {}
    ):

    # check if an element in operational limit is inf
    if "inf" in operational_limit.values():
        operational_limit = {k: np.inf if v == "inf" else v for k, v in operational_limit.items()}
    if "inf" in operational_limit_comp.values():
        operational_limit_comp = {k: np.inf if v == "inf" else v for k, v in operational_limit_comp.items()}

    if logger == None:
        logger, log_dir = create_folders_logger(single_run, label="use_case_")
    else:
        log_dir = os.path.join(log_dir, run_label)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    logger.info("Configuration variables of interest:")
    logger.info(f"Single run: {single_run}")
    if single_run == False:
        logger.info(f"Number of runs: {num_runs}")
    logger.info(f"Use all variables: {all_variables}")
    logger.info(f"Use case with new test: {use_case_new_test}")
    logger.info(f"PE method: {cfg['rel_point_cond_mut_info']}")
 

    logger.info("Reading the network file...")
    if net == None:
        net = pysmile.Network()
        if use_case_new_test == True:
            file_location = "outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestTrue/decision_models/DM_screening_rel_point_cond_mut_info_linear_new_test.xdsl"
        elif from_elicitation == True:
            file_location = "outputs/linear_rel_point_cond_mut_info_elicitTrue_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl"
        else:
            file_location = "outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl"
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
        logger.info("All variables are kept in the dataframe for calculation of utilities.")
        pass


    

    if use_case_new_test == True:
        operational_limit["New_test"] = 20000
        operational_limit["Stool_DNA"] = 0
        operational_limit["FIT"] = 20000

    logger.info(f"Operational limits for the screening strategies: {operational_limit}")


    if single_run:
        logger.info("A single simulation of the tests will be performed...")

        df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test, logger=logger, full_calculation = True)
        df_test_for_new_str_w_lim = df_test.copy()
        df_test_for_old_str = df_test.copy()
        plot_screening_counts(counts, possible_outcomes, operational_limit, log_dir=log_dir)
        logger.info("Calculation finished!")


        logger.info("----------------------")
        logger.info("New screening strategy without operational limits")
        df_test, total_cost, time_taken, positive_predictions_counts = new_screening_strategy(df_test, net, possible_outcomes,  counts, limit = False,  logger=logger, verbose = True)
        counts_best_opt = df_test["best_option"].value_counts()
        counts_best_opt = counts_best_opt.reindex(possible_outcomes, fill_value = 0)

        num_participants = df_test.shape[0] - counts_best_opt["No_scr_no_col"]

        logger.info(f"---> Total cost of the strategy: {total_cost:.2f} €")
        logger.info(f"---> Mean cost per screened participant: {total_cost/num_participants:.2f} €")
        logger.info(f"---> Mean cost per individual in the total population: {total_cost/df_test.shape[0]:.2f} €")
        logger.info(f"---> Total time for the simulation: {time_taken:.2f} seconds")

        y_true_new = df_test["CRC"]
        y_pred_new = df_test["Final_decision"]

        counts_new = df_test.groupby(["best_option", "Prediction_screening", "Prediction_colonoscopy", "Final_decision","CRC"])[["CRC"]].count()
        counts_new.to_csv(f"{log_dir}/counts_new.csv")
        logger.info(f"---> Distribution of positive predictions: \n {counts_new}")
        
        report, conf_matrix = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost, label = f"new_strategy_{run_label}", log_dir = log_dir)
        logger.info(report)


        logger.info("----------------------")
        logger.info("New screening strategy with operational limits")
        df_test_for_new_str_w_lim_util, total_cost_w_lim, time_taken_w_lim, positive_prediction_counts = new_screening_strategy(df_test_for_new_str_w_lim, net, possible_outcomes, counts, limit = True, operational_limit = operational_limit,  logger=logger, verbose = True)
        counts_best_opt_w_lim = df_test_for_new_str_w_lim_util["best_option_w_lim"].value_counts()
        counts_best_opt_w_lim = counts_best_opt_w_lim.reindex(possible_outcomes, fill_value = 0)
        num_participants_new_lim = df_test_for_new_str_w_lim_util.shape[0] - counts_best_opt_w_lim["No_scr_no_col"]

        logger.info(f"---> Total cost of the strategy: {total_cost_w_lim:.2f} €")
        logger.info(f"---> Mean cost per screened participant: {total_cost_w_lim/num_participants_new_lim:.2f} €")
        logger.info(f"---> Mean cost per individual in the total population: {total_cost_w_lim/df_test.shape[0]:.2f} €")
        logger.info(f"---> Total time for the simulation: {time_taken_w_lim:.2f} seconds")

        y_true_new = df_test_for_new_str_w_lim_util["CRC"]
        y_pred_new = df_test_for_new_str_w_lim_util["Final_decision"]

        counts_new_str_w_lim = df_test_for_new_str_w_lim_util.groupby(["best_option_w_lim", "Prediction_screening", "Prediction_colonoscopy", "Final_decision", "CRC"])[["CRC"]].count()
        counts_new_str_w_lim.to_csv(f"{log_dir}/counts_new_w_lim.csv")
        logger.info(f"---> Distribution of positive predictions: \n {counts_new_str_w_lim}")

        report, conf_matrix = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost_w_lim,  label = f"new_strategy_with_limits_{run_label}", log_dir = log_dir)
        logger.info(report)


        logger.info("----------------------")
        logger.info("Old screening strategy")
        df_test_for_old_str, total_cost_old, time_taken_old = old_screening_strategy(df_test_for_old_str, net, possible_outcomes, logger=logger, verbose = True)
        counts_best_opt_old = df_test_for_old_str["best_option"].value_counts()
        counts_best_opt_old = counts_best_opt_old.reindex(possible_outcomes, fill_value = 0)
        num_participants_old = df_test_for_old_str.shape[0] - counts_best_opt_old["No_scr_no_col"]

        logger.info(f"---> Total cost of the strategy: {total_cost_old:.2f} €")
        logger.info(f"---> Mean cost per screened participant: {total_cost_old/num_participants_old:.2f} €")
        logger.info(f"---> Mean cost per individual in the total population: {total_cost_old/df_test.shape[0]:.2f} €")
        logger.info(f"---> Total time for the simulation: {time_taken_old:.2f} seconds")

        y_true_old = df_test_for_old_str["CRC"]
        y_pred_old = df_test_for_old_str["Final_decision"]

        counts_old = df_test_for_old_str.groupby(["best_option", "Prediction_screening", "Prediction_colonoscopy", "Final_decision", "CRC"])[["CRC"]].count()
        counts_old.to_csv(f"{log_dir}/counts_old.csv")
        logger.info(f"---> Distribution of positive predictions: \n {counts_old}")

        report, conf_matrix = plot_classification_results(y_true_old, y_pred_old, total_cost = total_cost_old, label = f"old_strategy_{run_label}", log_dir= log_dir)
        logger.info(report)


        logger.info("Comparison of the strategies")
        
        if use_case_new_test == True:
            operational_limit_comp["New_test"] = 0

        df_test_for_new_str_w_lim_util, total_cost_w_lim, time_taken_w_lim, positive_prediction_counts = new_screening_strategy(df_test_for_new_str_w_lim, net, possible_outcomes, counts, limit = True, operational_limit = operational_limit_comp,  logger=logger, verbose = True)
        counts_best_opt_w_lim = df_test_for_new_str_w_lim_util["best_option_w_lim"].value_counts()
        counts_best_opt_w_lim = counts_best_opt_w_lim.reindex(possible_outcomes, fill_value = 0)
        num_participants_new_lim = df_test_for_new_str_w_lim_util.shape[0] - counts_best_opt_w_lim["No_scr_no_col"]

        logger.info(f"---> Total cost of the strategy: {total_cost_w_lim:.2f} €")
        logger.info(f"---> Mean cost per screened participant: {total_cost_w_lim/num_participants_new_lim:.2f} €")
        logger.info(f"---> Mean cost per individual in the total population: {total_cost_w_lim/df_test.shape[0]:.2f} €")
        logger.info(f"---> Total time for the simulation: {time_taken_w_lim:.2f} seconds")

        y_true_new = df_test_for_new_str_w_lim_util["CRC"]
        y_pred_new = df_test_for_new_str_w_lim_util["Final_decision"]

        counts_new_str_w_lim = df_test_for_new_str_w_lim_util.groupby(["best_option_w_lim", "Prediction_screening", "Prediction_colonoscopy", "Final_decision", "CRC"])[["CRC"]].count()
        counts_new_str_w_lim.to_csv(f"{log_dir}/counts_new_w_lim.csv")
        logger.info(f"---> Distribution of positive predictions: \n {counts_new_str_w_lim}")

        report, conf_matrix = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost_w_lim,  label = f"new_strategy_with_limits_{run_label}", log_dir = log_dir)
        logger.info(report)



    else:
        logger.info("Multiple simulations of the tests will be performed...")

        report_df_new = []
        report_df_new_w_lim = []
        report_df_old = []
        report_df_comp = []

        conf_matrix_new_list = []
        conf_matrix_new_w_lim_list = []
        conf_matrix_old_list = []
        conf_matrix_comp_list = []

        total_cost_list_old = []
        total_cost_list_new =[]
        total_cost_list_new_w_lim = []
        total_cost_list_comp = []


        df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test)
        plot_screening_counts(counts, possible_outcomes, operational_limit, log_dir=log_dir)
        
        with ProcessPoolExecutor(max_workers=cfg['max_workers']) as executor:
            futures = [executor.submit(run_experiment, i, df_test, file_location, possible_outcomes, counts, operational_limit, use_case_new_test, log_dir) for i in range(num_runs)]
            all_results = []
            for future in tqdm(as_completed(futures), total=num_runs, desc="Processing iterations"):
                all_results.append(future.result())




        report_df_new = [result["report_df_new"] for result in all_results]
        conf_matrix_new_list = [result["conf_matrix_new"] for result in all_results]
        total_cost_list_new = [result["total_cost_new"] for result in all_results]
        
        report_df_new = pd.concat(report_df_new, axis = 0, keys=range(len(report_df_new)))
        
        mean_conf_matrix_new = np.stack(conf_matrix_new_list, axis = 0).mean(axis = 0)
        std_conf_matrix_new = np.stack(conf_matrix_new_list, axis = 0).std(axis = 0)
        
        mean_report_new = report_df_new.groupby(level=1, sort = False).mean()
        std_report_new = report_df_new.groupby(level=1, sort = False).std()

        mean_cost_new = np.array(total_cost_list_new).mean()
        std_cost_new = np.array(total_cost_list_new).std()

        plot_estimations_w_error_bars(mean_report_new, std_report_new, label="new_strategy", log_dir = log_dir)
        plot_classification_results(report_df = mean_report_new, conf_matrix = mean_conf_matrix_new, std_conf_matrix = std_conf_matrix_new, total_cost=mean_cost_new, label = f"mean_new_strategy_{run_label}", plot= True, log_dir = log_dir)

        
        report_df_new_w_lim = [result["report_df_new_w_lim"] for result in all_results]
        conf_matrix_new_w_lim_list = [result["conf_matrix_new_w_lim"] for result in all_results]
        total_cost_list_new_w_lim = [result["total_cost_new_w_lim"] for result in all_results]
        

        report_df_new_w_lim = pd.concat(report_df_new_w_lim, axis = 0, keys=range(len(report_df_new_w_lim)))
        
        mean_conf_matrix_new_w_lim = np.stack(conf_matrix_new_w_lim_list, axis = 0).mean(axis = 0)
        std_conf_matrix_new_w_lim = np.stack(conf_matrix_new_w_lim_list, axis = 0).std(axis = 0)
        
        mean_report_new_w_lim = report_df_new_w_lim.groupby(level=1, sort = False).mean()
        std_report_new_w_lim = report_df_new_w_lim.groupby(level=1, sort = False).std()

        mean_cost_new_w_lim = np.array(total_cost_list_new_w_lim).mean()
        std_cost_new_w_lim = np.array(total_cost_list_new_w_lim).std()


        plot_estimations_w_error_bars(mean_report_new_w_lim, std_report_new_w_lim, label="new_strategy_with_limits", log_dir=log_dir)
        plot_classification_results(report_df = mean_report_new_w_lim, conf_matrix=mean_conf_matrix_new_w_lim, std_conf_matrix=std_conf_matrix_new_w_lim, total_cost=mean_cost_new_w_lim, label = f"mean_new_strategy_with_limits_{run_label}", plot= True, log_dir = log_dir)

        
        report_df_old = [result["report_df_old"] for result in all_results]
        conf_matrix_old_list = [result["conf_matrix_old"] for result in all_results]
        total_cost_list_old = [result["total_cost_old"] for result in all_results]

        report_df_old = pd.concat(report_df_old, axis = 0, keys=range(len(report_df_old)))
        
        mean_conf_matrix_old = np.stack(conf_matrix_old_list, axis = 0).mean(axis = 0)
        std_conf_matrix_old = np.stack(conf_matrix_old_list, axis = 0).std(axis = 0)
        
        mean_report_old = report_df_old.groupby(level=1, sort = False).mean()
        std_report_old = report_df_old.groupby(level=1, sort = False).std()

        mean_cost_old = np.array(total_cost_list_old).mean()
        std_cost_old = np.array(total_cost_list_old).std()

        plot_estimations_w_error_bars(mean_report_old, std_report_old, label="old_strategy", log_dir = log_dir)
        report_df_old, conf_matrix_old = plot_classification_results(report_df = mean_report_old, total_cost = mean_cost_old, conf_matrix= mean_conf_matrix_old, std_conf_matrix= std_conf_matrix_old, label = f"mean_old_strategy_{run_label}", plot= True, log_dir = log_dir)

        report_df_comp = [result["report_df_comp"] for result in all_results]
        conf_matrix_comp_list = [result["conf_matrix_comp"] for result in all_results]
        total_cost_list_comp = [result["total_cost_comp"] for result in all_results]

        report_df_comp = pd.concat(report_df_comp, axis = 0, keys=range(len(report_df_comp)))

        mean_conf_matrix_comp = np.stack(conf_matrix_comp_list, axis = 0).mean(axis = 0)
        std_conf_matrix_comp = np.stack(conf_matrix_comp_list, axis = 0).std(axis = 0)

        mean_report_comp = report_df_comp.groupby(level=1, sort = False).mean()
        std_report_comp = report_df_comp.groupby(level=1, sort = False).std()

        mean_cost_comp = np.array(total_cost_list_comp).mean()
        std_cost_comp = np.array(total_cost_list_comp).std()

        plot_estimations_w_error_bars(mean_report_comp, std_report_comp, label="new_strategy_comparison", log_dir=log_dir)
        report_df_comp, conf_matrix_comp = plot_classification_results(report_df = mean_report_comp, conf_matrix=mean_conf_matrix_comp, std_conf_matrix=std_conf_matrix_comp, total_cost=mean_cost_comp, label = f"mean_new_strategy_comparison_{run_label}", plot= True, log_dir = log_dir)

        # save f1 score for the positive class 
        pdb.set_trace()
        best_f1_score[run_label]["old"] = report_df_old.loc["Positive"]["f1-score"]
        best_f1_score[run_label]["comparison"] = report_df_comp.loc["Positive"]["f1-score"]

        return best_f1_score

    





def run_experiment(i, df_test, file_location, possible_outcomes, counts, operational_limit, use_case_new_test, log_dir):

    results = {
        "report_df_new": None,
        "conf_matrix_new": None,
        "total_cost_new": None,
        "report_df_new_w_lim": None,
        "conf_matrix_new_w_lim": None,
        "total_cost_new_w_lim": None,
        "report_df_old": None,
        "conf_matrix_old": None,
        "total_cost_old": None,
        "report_df_comp": None,
        "conf_matrix_comp": None,
        "total_cost_comp": None
    }

    net = pysmile.Network()
    net.read_file(file_location)

    df_test_new = df_test.copy()
    df_test_new_w_lim = df_test.copy()
    df_test_old = df_test.copy()
    df_test_comp = df_test.copy()
    

    df_test_new, total_cost_new, time_taken, positive_predictions_count = new_screening_strategy(df_test_new, net, possible_outcomes, counts = counts, limit=False, operational_limit = dict(zip(operational_limit.keys(), counts)))

    y_true_new = df_test_new["CRC"]
    y_pred_new = df_test_new["Final_decision"]
    report_new, conf_matrix_new = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost_new, label = "new_strategy", plot = False, log_dir = log_dir)

    results["report_df_new"] = report_new
    results["conf_matrix_new"] = conf_matrix_new
    results["total_cost_new"] = total_cost_new


    df_test_new_w_lim, total_cost_new_w_lim, time_taken, positive_predictions_count = new_screening_strategy(df_test_new_w_lim, net, possible_outcomes, counts=counts, limit=True, operational_limit = operational_limit)

    y_true_new_w_lim = df_test_new_w_lim["CRC"]
    y_pred_new_w_lim = df_test_new_w_lim["Final_decision"]
    report_new_w_lim, conf_matrix_new_w_lim = plot_classification_results(y_true_new_w_lim, y_pred_new_w_lim, total_cost=total_cost_new_w_lim, label = "new_strategy_with_limits", plot = False)

    results["report_df_new_w_lim"] = report_new_w_lim
    results["conf_matrix_new_w_lim"] = conf_matrix_new_w_lim
    results["total_cost_new_w_lim"] = total_cost_new_w_lim


    df_test_old, total_cost_old, time_taken = old_screening_strategy(df_test_old, net, possible_outcomes)

    y_true_old = df_test_old["CRC"]
    y_pred_old = df_test_old["Final_decision"]
    report_old, conf_matrix_old = plot_classification_results(y_true_old, y_pred_old, total_cost = total_cost_old, label = "old_strategy", plot = False, log_dir = log_dir)

    results["report_df_old"] = report_old
    results["conf_matrix_old"] = conf_matrix_old
    results["total_cost_old"] = total_cost_old

    # logger.info("Comparison of the strategies")
    operational_limit_comp = { "No_scr_no_col": np.inf, "No_scr_col": 0, "gFOBT": 0,"FIT": 49074, 
                        "Blood_based": 0,"Stool_DNA": 0, "CTC": 0, "Colon_capsule": 0,
    }
    if use_case_new_test == True:
        operational_limit_comp["New_test"] = 0

    df_test_comp_util, total_cost_comp, time_taken_w_lim, positive_prediction_counts = new_screening_strategy(df_test_comp, net, possible_outcomes, counts, limit = True, operational_limit = operational_limit_comp)
    
    y_true_new = df_test_comp_util["CRC"]
    y_pred_new = df_test_comp_util["Final_decision"]
    report_comp, conf_matrix_comp = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost_comp,  label = "new_strategy_with_limits", log_dir = log_dir, plot = False)
    
    results["report_df_comp"] = report_comp
    results["conf_matrix_comp"] = conf_matrix_comp
    results["total_cost_comp"] = total_cost_comp

    
    return results




if __name__ == "__main__":
    use_case_new_strategy()