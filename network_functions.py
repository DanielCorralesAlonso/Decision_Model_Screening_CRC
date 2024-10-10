
import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from simulations import simulate_test_results
import time

from preprocessing import preprocessing

np.seterr(divide='ignore', invalid = 'ignore')


def calculate_network_utilities(net, df_test, full_calculation = False):

    # if not full_calculation:

    #     df_utilities_2016 = pd.read_csv("outputs/utilities_2016_fulldataset.csv")

    #     possible_outcomes = net.get_outcome_ids("Screening")
    #     possible_outcomes.insert(0, "No_scr_no_col")
    #     possible_outcomes[1] = "No_scr_col"

    #     counts = np.zeros(len(possible_outcomes))
    #     all_values = []
    #     best_options = []

    #     for i in range(df_utilities_2016.shape[0]):
    #         max_index = np.argmax(df_utilities_2016.iloc[i].values[:-1])
    #         best_options.append(possible_outcomes[max_index])

    #         counts[max_index] += 1

    #     best_options_df = pd.DataFrame(best_options, columns = ["Best_option"])

    # else:

    print("Full calculation of utilities...")

    net = pysmile.Network()
    net.read_file(f"outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl")

    y = np.array(list(df_test["CRC"]*1))

    possible_outcomes = net.get_outcome_ids("Screening")
    possible_outcomes.insert(0, "No_scr_no_col")
    possible_outcomes[1] = "No_scr_col"

    counts = np.zeros(len(possible_outcomes))
    all_values = []
    best_options = []

    df_utilities_2016 = pd.DataFrame([], columns = possible_outcomes)

    # 1. Identify unique rows
    unique_rows = df_test.drop(columns="CRC").copy()
    unique_rows = unique_rows.drop_duplicates().copy()

    # 2. Define a function to calculate the utility for a given row
    def calculate_utility(row):
        sample = row.to_dict()
        net.clear_all_evidence()

        for key, value in sample.items():
            if type(value) == np.bool_:
                net.set_evidence(key, bool(value))
            else:
                net.set_evidence(key, value)

        net.update_beliefs()

        utilities = net.get_node_value("Screening") 
        utilities.insert(1, net.get_node_value("Colonoscopy")[1])

        return np.array(utilities)
    
    def argmax_utility(row):
        x = row["utilities"]
        return possible_outcomes[np.argmax(x)]


    # 3. Apply the function to the unique rows and store the result in a new column
    unique_rows['utilities'] = unique_rows.apply(calculate_utility, axis=1)
    unique_rows['best_option'] = unique_rows.apply(argmax_utility, axis = 1)

    # 4. Map these calculated values back to the original DataFrame
    # Using a merge or map-like operation to match the calculated values to duplicates
    df_with_calculated = df_test.merge(unique_rows, on=list(df_test.drop(columns="CRC").columns), how='left')

    utilities_series = pd.Series(df_with_calculated['utilities'].values)

    # 1. Convert each element (list of lists) into a 2D numpy array
    utilities_2d_list = [np.array(sublist) for sublist in utilities_series]

    # 2. Flatten into a DataFrame with a MultiIndex for rows and columns
    # Assuming each sublist has a shape (m, n), we can concatenate them along axis=0
    df_utilities_2016 = pd.DataFrame(np.vstack(utilities_2d_list), columns = possible_outcomes)
    df_utilities_2016["best_option"] = df_with_calculated["best_option"]

    # best_options = df_utilities_2016.apply(lambda x: possible_outcomes[np.argmax(x)], axis=1)
    counts = df_utilities_2016["best_option"].value_counts()
    counts = counts.reindex(possible_outcomes, fill_value=0)

    # best_options_df = pd.DataFrame(best_options, columns = ["Best_option"])

    df_test = df_test.merge(df_utilities_2016, left_index=True, right_index=True)

    return df_test, counts, possible_outcomes



def new_screening_strategy(df_test, net, possible_outcomes):
    t1 = time.time()

    df_test["Colonoscopy"] = None

    for i, strategy in enumerate(possible_outcomes):
        
        if strategy == "No_scr_no_col":
            selected_patients = df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index].copy()

            y_selected_patients = selected_patients["CRC"]

            df_test.loc[selected_patients.index, "Prediction_screening"] = 0
            df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = 0
            df_test.loc[selected_patients.index, "Final_decision"] = 0
            continue

        elif strategy == "No_scr_col":
            selected_patients = df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index].copy()

            y_selected_patients = selected_patients["CRC"]

            df_test.loc[selected_patients.index, "Prediction_screening"] = 0

            spec_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[0, 1 ,1]
            sens_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[1, 1 ,2]

            df_col = simulate_test_results(sens_col, spec_col, y_selected_patients)

            df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = df_col["TestResult"]
            df_test.loc[selected_patients.index, "Final_decision"] = df_col["TestResult"]
            continue

        else:

            selected_patients = df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index].copy()

            y_selected_patients = selected_patients["CRC"]

            spec_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)[0, i-1, 1]
            sens_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)[1, i-1, 2]

            df_scr = simulate_test_results(sens_strategy, spec_strategy, y_selected_patients) 

            df_test.loc[selected_patients.index, "Prediction_screening"] = df_scr["TestResult"]
            df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = 0
            df_test.loc[selected_patients.index, "Final_decision"] = df_scr["TestResult"]

            # For FIT positives, perform colonoscopy:
            FIT_positives = df_scr[df_scr["TestResult"] == 1]
            y_FIT_positives = FIT_positives["Condition"]

            spec_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[0, 1 ,1]
            sens_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[1, 1 ,2]
            
            df_col = simulate_test_results(sens_col, spec_col, y_FIT_positives)

            df_test.loc[FIT_positives.index, "Colonoscopy"] = "Colonoscopy"
            df_test.loc[FIT_positives.index, "Prediction_colonoscopy"] = df_col["TestResult"]
            df_test.loc[FIT_positives.index, "Final_decision"] = df_col["TestResult"]


    net.update_beliefs()
    total_cost_of_screening = 0.0
    total_cost_of_colonoscopy = 0.0
    scr_costs = net.get_node_value("COST")[::2]
    scr_costs.insert(1, 0.0)

    col_costs = net.get_node_value("COST")[1]

    counts = df_test["best_option"].value_counts()
    counts = counts.reindex(possible_outcomes, fill_value=0)

    total_cost_of_screening = np.matmul(np.array(scr_costs),counts.values)
    total_cost_of_colonoscopy = col_costs*(df_test["Colonoscopy"].value_counts()["Colonoscopy"])


    total_cost = total_cost_of_screening + total_cost_of_colonoscopy
    t2 = time.time()
    time_taken = t2 - t1


    return df_test, total_cost, time_taken



def old_screening_strategy(df_test, net, possible_outcomes):

    t1 = time.time()

    df_test["Screening_strategy"] = "No_scr_no_col"
    df_test["Final_decision"] = 0
    df_test.loc[df_test["Age"] == "age_5_old_adult","Screening_strategy"] = "FIT"
    df_test["Colonoscopy"] = None

    selected_patients = df_test[df_test["Age"] == "age_5_old_adult"].copy()
    y_selected_patients = selected_patients["CRC"]

    # Do FIT to all patients over 50
    spec_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)[0, 2, 1]
    sens_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)[1, 2, 2]

    df_scr = simulate_test_results(sens_strategy, spec_strategy, y_selected_patients) 

    df_test.loc[selected_patients.index, "Prediction_screening"] = df_scr["TestResult"]
    df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = 0
    df_test.loc[selected_patients.index, "Final_decision"] = df_scr["TestResult"]

    # For FIT positives, perform colonoscopy:
    FIT_positives = df_scr[df_scr["TestResult"] == 1]
    y_FIT_positives = FIT_positives["Condition"]

    spec_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[0, 1 ,1]
    sens_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[1, 1 ,2]

    df_col = simulate_test_results(sens_col, spec_col, y_FIT_positives)

    df_test.loc[FIT_positives.index, "Colonoscopy"] = "Colonoscopy"
    df_test.loc[FIT_positives.index, "Prediction_colonoscopy"] = df_col["TestResult"]
    df_test.loc[FIT_positives.index, "Final_decision"] = df_col["TestResult"]


    total_cost_of_screening = 0.0
    total_cost_of_colonoscopy = 0.0

    net.update_beliefs()
    scr_costs = net.get_node_value("COST")[::2]
    scr_costs.insert(1, 0.0)

    col_costs = net.get_node_value("COST")[1]

    counts = df_test["best_option"].value_counts()
    counts = counts.reindex(possible_outcomes, fill_value=0)

    total_cost_of_screening = np.matmul(np.array(scr_costs),counts.values)
    total_cost_of_colonoscopy = col_costs*(df_test["Colonoscopy"].value_counts()["Colonoscopy"])

    total_cost = total_cost_of_screening + total_cost_of_colonoscopy
    t2 = time.time()
    time_taken = t2 - t1


    return df_test, total_cost, time_taken