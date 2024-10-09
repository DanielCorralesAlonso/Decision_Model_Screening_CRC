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


single_run = True
num_runs = 20

if single_run:
    print("A single simulation of the tests will be performed...")

    print("Trying to load utilities from previous calculations...")
    try: 
        best_options_df, df_utilities_2016, counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = False)
        # plot_screening_counts(counts, possible_outcomes)
        print("Success!")
    except:
        print("Failed!")
        print("Full calculation of utilies for each patient will be required. This may take a while...")
        best_options_df, df_utilities_2016, counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = True)
        plot_screening_counts(counts, possible_outcomes)
        print("Calculation finished!")

    print("----------------------")
    print("New screening strategy")
    df_test_extended, total_cost, time_taken = new_screening_strategy(df_test, net, best_options_df, possible_outcomes)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true = df_test_extended["CRC"]
    y_pred = df_test_extended["Final_decision"]
    report = plot_classification_results(y_true, y_pred, label = "new_strategy")
    print(report)

    print("----------------------")
    print("Old screening strategy")
    df_test_extended, total_cost, time_taken = old_screening_strategy(df_test, net, best_options_df, possible_outcomes)

    print(f"---> Total cost of the strategy: {total_cost:.2f} €")
    print(f"---> Total time for the simulation: {time_taken:.2f} seconds")

    y_true = df_test_extended["CRC"]
    y_pred = df_test_extended["Final_decision"]
    report = plot_classification_results(y_true, y_pred, label = "old_strategy")
    print(report)

else:
    print("Multiple simulations of the tests will be performed...")

    report_df_new = []
    report_df_old = []

    for i in range(num_runs):

        try: 
            best_options_df, df_utilities_2016, counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = False)
        except:
            best_options_df, df_utilities_2016, counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = True)
    
        df_test_extended, total_cost, time_taken = new_screening_strategy(df_test, net, best_options_df, possible_outcomes)

        y_true = df_test_extended["CRC"]
        y_pred = df_test_extended["Final_decision"]
        report = plot_classification_results(y_true, y_pred, label = "new_strategy")

        report_df_new.append(report[:-1])

        df_test_extended, total_cost, time_taken = old_screening_strategy(df_test, net, best_options_df, possible_outcomes)

        y_true = df_test_extended["CRC"]
        y_pred = df_test_extended["Final_decision"]
        report = plot_classification_results(y_true, y_pred, label = "old_strategy")

        report_df_old.append(report[:-1])
    
    report_df_new = pd.concat(report_df_new, axis = 0, keys=range(len(report_df_new)))
    mean_report_new = report_df_new.groupby(level=1, sort = False).mean()
    std_report_new = report_df_new.groupby(level=1, sort = False).std()
    SE_report_new = std_report_new / np.sqrt(num_runs)

    plot_estimations_w_error_bars(mean_report_new, std_report_new, SE_report_new, label="New strategy")


    report_df_old = pd.concat(report_df_old, axis = 0, keys=range(len(report_df_old)))
    mean_report_old = report_df_old.groupby(level=1, sort = False).mean()
    std_report_old = report_df_old.groupby(level=1, sort = False).std()
    SE_report_old = std_report_old / np.sqrt(num_runs)

    plot_estimations_w_error_bars(mean_report_old, std_report_old, SE_report_old, label="Old strategy")

    

    






