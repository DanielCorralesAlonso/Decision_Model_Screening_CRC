import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from network_functions import calculate_network_utilities, new_screening_strategy, old_screening_strategy
from simulations import plot_classification_results
from plots import plot_estimations_w_error_bars, plot_screening_counts

from preprocessing import preprocessing
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid = 'ignore')

net = pysmile.Network()
net.read_file(f"outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl")


df_test = pd.read_csv("private/df_2016.csv")
df_test = preprocessing(df_test)
df_test = df_test.rename(columns = {"Hyperchol.": "Hyperchol_"})
# Just keep variables that influence the decision
df_test.drop(columns = ["Hyperchol_", "Hypertension", "Diabetes", "SES", "Anxiety", "Depression"], inplace = True)

# df_test = df_test.sample(10000, random_state = 42).reset_index(drop=True)


single_run = False
num_runs = 10

if single_run:
    print("A single simulation of the tests will be performed...")

    df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = True)
    df_test_for_old_str = df_test.copy()
    plot_screening_counts(counts, possible_outcomes)
    print("Calculation finished!")



    print("----------------------")
    print("New screening strategy")
    df_test, total_cost, time_taken = new_screening_strategy(df_test, net, possible_outcomes)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true_new = df_test["CRC"]
    y_pred_new = df_test["Final_decision"]
    report = plot_classification_results(y_true_new, y_pred_new, label = "new_strategy")
    print(report)



    print("----------------------")
    print("Old screening strategy")
    df_test_for_old_str, total_cost, time_taken = old_screening_strategy(df_test_for_old_str, net, possible_outcomes)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true_old = df_test_for_old_str["CRC"]
    y_pred_old = df_test_for_old_str["Final_decision"]
    report = plot_classification_results(y_true_old, y_pred_old, label = "old_strategy")
    print(report)

else:
    print("Multiple simulations of the tests will be performed...")

    report_df_new = []
    report_df_old = []
    total_cost_list_old = []
    total_cost_list_new =[]

    df_test, counts, possible_outcomes = calculate_network_utilities(net, df_test)

    for i in range(num_runs):
        df_test_new = df_test.copy()
        df_test_old = df_test.copy()
        

        df_test_new, total_cost_new, time_taken = new_screening_strategy(df_test_new, net, possible_outcomes)

        y_true_new = df_test_new["CRC"]
        y_pred_new = df_test_new["Final_decision"]
        report_new = plot_classification_results(y_true_new, y_pred_new, label = "new_strategy")

        report_df_new.append(report_new[:-1])
        total_cost_list_new.append(total_cost_new)


        df_test_old, total_cost_old, time_taken = old_screening_strategy(df_test_old, net, possible_outcomes)

        y_true_old = df_test_old["CRC"]
        y_pred_old = df_test_old["Final_decision"]
        report_old = plot_classification_results(y_true_old, y_pred_old, label = "old_strategy")

        report_df_old.append(report_old[:-1])
        total_cost_list_old.append(total_cost_old)

        print(f"Run {i} completed!")

    
    report_df_new = pd.concat(report_df_new, axis = 0, keys=range(len(report_df_new)))
    mean_report_new = report_df_new.groupby(level=1, sort = False).mean()
    std_report_new = report_df_new.groupby(level=1, sort = False).std()
    SE_report_new = std_report_new / np.sqrt(num_runs)
    plot_estimations_w_error_bars(mean_report_new, std_report_new, SE_report_new, label="New strategy")
    
    mean_cost_new = np.array(total_cost_list_new).mean()
    std_cost_new = np.array(total_cost_list_new).std()
    print(f"Average cost: {mean_cost_new:.2f} (+/- {std_cost_new:.2f})  €")



    report_df_old = pd.concat(report_df_old, axis = 0, keys=range(len(report_df_old)))
    mean_report_old = report_df_old.groupby(level=1, sort = False).mean()
    std_report_old = report_df_old.groupby(level=1, sort = False).std()
    SE_report_old = std_report_old / np.sqrt(num_runs)
    plot_estimations_w_error_bars(mean_report_old, std_report_old, SE_report_old, label="Old strategy")

    mean_cost_old = np.array(total_cost_list_old).mean()
    std_cost_old = np.array(total_cost_list_old).std()
    print(f"Average cost: {mean_cost_old:.2f} (+/- {std_cost_old:.2f})  €")

    






