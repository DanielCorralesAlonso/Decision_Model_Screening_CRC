import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from network_functions import calculate_network_utilities, new_screening_strategy, old_screening_strategy
from simulations import plot_classification_results
from plots import plot_estimations_w_error_bars, plot_screening_counts

from preprocessing import preprocessing
import matplotlib.pyplot as plt

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

import argparse
import pdb

np.seterr(divide='ignore', invalid = 'ignore')

parser = argparse.ArgumentParser(description="Update the influence diagram")
    
# Introduction of arguments
parser.add_argument('--single_run', type=str, default=cfg["single_run"],)
parser.add_argument('--num_runs', type=int, default= cfg["num_runs"],)

# Parse the arguments
args = parser.parse_args()

net = pysmile.Network()
net.read_file(f"outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl")


df_test = pd.read_csv("private/df_2016.csv")
df_test = preprocessing(df_test)
df_test = df_test.rename(columns = {"Hyperchol.": "Hyperchol_"})
# Just keep variables that influence the decision
df_test.drop(columns = ["Hyperchol_", "Hypertension", "Diabetes", "SES", "Anxiety", "Depression"], inplace = True)

# df_test = df_test.sample(10000, random_state = 42).reset_index(drop=True)


operational_limit = {
    "No_scr_no_col": np.inf,
    "No_scr_col": 30000,
    "gFOBT": 30000,
    "FIT": 30000,
    "Blood_based": 30000,
    "Stool_DNA": 30000,
    "CTC": 30000,
    "Colon_capsule": 30000,
}

#transform operational limits to df with one column
# operational_limit = pd.DataFrame(operational_limit, index = [0])


single_run = bool(args.single_run)
num_runs = int(args.num_runs)


if single_run:
    print("A single simulation of the tests will be performed...")

    df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = True)
    df_test_for_new_str_w_lim = df_test.copy()
    df_test_for_old_str = df_test.copy()
    plot_screening_counts(counts, possible_outcomes, operational_limit.values())
    print("Calculation finished!")


    print("----------------------")
    print("New screening strategy without operational limits")
    df_test, total_cost, time_taken, positive_predictions_counts = new_screening_strategy(df_test, net, possible_outcomes,  counts, limit = False, verbose = True)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Mean cost per patient: {total_cost/df_test.shape[0]:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true_new = df_test["CRC"]
    y_pred_new = df_test["Final_decision"]

    counts_new = df_test.groupby(["best_option", "Prediction_screening", "Prediction_colonoscopy"])[["CRC"]].sum()
    print(f"---> Distribution of positive predictions: \n {counts_new}")
    
    report, conf_matrix = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost, label = "new_strategy")
    print(report)


    print("----------------------")
    print("New screening strategy with operational limits")
    df_test_for_new_str_w_lim, total_cost, time_taken, positive_prediction_counts = new_screening_strategy(df_test_for_new_str_w_lim, net, possible_outcomes, counts, limit = True, operational_limit = operational_limit, verbose = True)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Mean cost per patient: {total_cost/df_test_for_new_str_w_lim.shape[0]:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true_new = df_test_for_new_str_w_lim["CRC"]
    y_pred_new = df_test_for_new_str_w_lim["Final_decision"]

    counts_new_str_w_lim = df_test_for_new_str_w_lim.groupby(["best_option", "Prediction_screening", "Prediction_colonoscopy"])[["CRC"]].sum()
    print(f"---> Distribution of positive predictions: \n {counts_new_str_w_lim}")

    report, conf_matrix = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost,  label = "new_strategy_with_limits")
    print(report)


    print("----------------------")
    print("Old screening strategy")
    df_test_for_old_str, total_cost, time_taken = old_screening_strategy(df_test_for_old_str, net, possible_outcomes, verbose = True)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Mean cost per patient: {total_cost/df_test_for_old_str.shape[0]:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true_old = df_test_for_old_str["CRC"]
    y_pred_old = df_test_for_old_str["Final_decision"]

    counts_old = df_test_for_old_str.groupby(["best_option", "Prediction_screening", "Prediction_colonoscopy"])[["CRC"]].sum()
    print(f"---> Distribution of positive predictions: \n {counts_old}")

    report, conf_matrix = plot_classification_results(y_true_old, y_pred_old, total_cost = total_cost, label = "old_strategy")
    print(report)

else:
    print("Multiple simulations of the tests will be performed...")

    report_df_new = []
    report_df_new_w_lim = []
    report_df_old = []

    conf_matrix_new_list = []
    conf_matrix_new_w_lim_list = []
    conf_matrix_old_list = []

    total_cost_list_old = []
    total_cost_list_new =[]
    total_cost_list_new_w_lim = []

    df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test)

    for i in range(num_runs):
        df_test_new = df_test.copy()
        df_test_new_w_lim = df_test.copy()
        df_test_old = df_test.copy()
        

        df_test_new, total_cost_new, time_taken = new_screening_strategy(df_test_new, net, possible_outcomes, counts = counts, limit=False, operational_limit = dict(zip(operational_limit.keys(), counts)))

        y_true_new = df_test_new["CRC"]
        y_pred_new = df_test_new["Final_decision"]
        report_new, conf_matrix_new = plot_classification_results(y_true_new, y_pred_new, total_cost = total_cost_new, label = "new_strategy", plot = False)

        report_df_new.append(report_new)
        conf_matrix_new_list.append(conf_matrix_new)
        total_cost_list_new.append(total_cost_new)


        df_test_new_w_lim, total_cost_new_w_lim, time_taken = new_screening_strategy(df_test_new_w_lim, net, possible_outcomes, counts=counts, limit=True, operational_limit = operational_limit)

        y_true_new_w_lim = df_test_new_w_lim["CRC"]
        y_pred_new_w_lim = df_test_new_w_lim["Final_decision"]
        report_new_w_lim, conf_matrix_new_w_lim = plot_classification_results(y_true_new_w_lim, y_pred_new_w_lim, total_cost=total_cost_new_w_lim, label = "new_strategy_with_limits", plot = False)

        report_df_new_w_lim.append(report_new_w_lim)
        conf_matrix_new_w_lim_list.append(conf_matrix_new_w_lim)
        total_cost_list_new_w_lim.append(total_cost_new_w_lim)


        df_test_old, total_cost_old, time_taken = old_screening_strategy(df_test_old, net, possible_outcomes)

        y_true_old = df_test_old["CRC"]
        y_pred_old = df_test_old["Final_decision"]
        report_old, conf_matrix_old = plot_classification_results(y_true_old, y_pred_old, total_cost = total_cost_old, label = "old_strategy", plot = False)

        report_df_old.append(report_old)
        conf_matrix_old_list.append(conf_matrix_old)
        total_cost_list_old.append(total_cost_old)

        print(f"Run {i} completed!")

    
    report_df_new = pd.concat(report_df_new, axis = 0, keys=range(len(report_df_new)))
    conf_matrix_new = np.stack(conf_matrix_new_list, axis = 0).mean(axis = 0)
    mean_report_new = report_df_new.groupby(level=1, sort = False).mean()
    std_report_new = report_df_new.groupby(level=1, sort = False).std()
    SE_report_new = std_report_new / np.sqrt(num_runs)

    mean_cost_new = np.array(total_cost_list_new).mean()
    std_cost_new = np.array(total_cost_list_new).std()
    '''print(f"Average cost: {mean_cost_new:.2f} (+/- {std_cost_new:.2f})  €")
    print("Average cost per patient: {:.2f} €".format(mean_cost_new/df_test.shape[0]))'''

    plot_estimations_w_error_bars(mean_report_new, std_report_new, SE_report_new, label="new_strategy")
    plot_classification_results(report_df = mean_report_new, conf_matrix = conf_matrix_new, total_cost=mean_cost_new, label = "mean_new_strategy", plot= True)

    
    report_df_new_w_lim = pd.concat(report_df_new_w_lim, axis = 0, keys=range(len(report_df_new_w_lim)))
    conf_matrix_new_w_lim = np.stack(conf_matrix_new_w_lim_list, axis = 0).mean(axis = 0)
    mean_report_new_w_lim = report_df_new_w_lim.groupby(level=1, sort = False).mean()
    std_report_new_w_lim = report_df_new_w_lim.groupby(level=1, sort = False).std()
    SE_report_new_w_lim = std_report_new_w_lim / np.sqrt(num_runs)

    mean_cost_new_w_lim = np.array(total_cost_list_new_w_lim).mean()
    std_cost_new_w_lim = np.array(total_cost_list_new_w_lim).std()
    '''print(f"Average cost: {mean_cost_new_w_lim:.2f} (+/- {std_cost_new_w_lim:.2f})  €")
    print("Average cost per patient: {:.2f} €".format(mean_cost_new_w_lim/df_test.shape[0]))'''

    plot_estimations_w_error_bars(mean_report_new_w_lim, std_report_new_w_lim, SE_report_new_w_lim, label="new_strategy_with_limits")
    plot_classification_results(report_df = mean_report_new_w_lim, conf_matrix=conf_matrix_new_w_lim, total_cost=mean_cost_new_w_lim, label = "mean_new_strategy_with_limits", plot= True)

    


    report_df_old = pd.concat(report_df_old, axis = 0, keys=range(len(report_df_old)))
    conf_matrix_old = np.stack(conf_matrix_old_list, axis = 0).mean(axis = 0)
    mean_report_old = report_df_old.groupby(level=1, sort = False).mean()
    std_report_old = report_df_old.groupby(level=1, sort = False).std()
    SE_report_old = std_report_old / np.sqrt(num_runs)
    mean_cost_old = np.array(total_cost_list_old).mean()
    std_cost_old = np.array(total_cost_list_old).std()
    '''print(f"Average cost: {mean_cost_old:.2f} (+/- {std_cost_old:.2f})  €")
    print("Average cost per patient: {:.2f} €".format(mean_cost_old/df_test.shape[0]))'''

    plot_estimations_w_error_bars(mean_report_old, std_report_old, SE_report_old, label="old_strategy")
    plot_classification_results(report_df = mean_report_old, total_cost = mean_cost_old, conf_matrix= conf_matrix_old, label = "mean_old_strategy", plot= True)

    

    






