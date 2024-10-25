
import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from simulations import simulate_test_results
import time

import pdb

import logging
import datetime 
import os


from preprocessing import preprocessing

np.seterr(divide='ignore', invalid = 'ignore')


def calculate_network_utilities(net, df_test, logger=None, full_calculation = False):
    # pdb.set_trace()

    #net = pysmile.Network()
    # net.read_file(f"outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl")

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
    
    def max_utility(row):
        x = row["utilities"]
        return np.max(x)
    
    def argmax_utility(row):
        x = row["utilities"]
        return possible_outcomes[np.argmax(x)]
    
    

    # 3. Apply the function to the unique rows and store the result in a new column
    unique_rows['utilities'] = unique_rows.apply(calculate_utility, axis=1)
    unique_rows['max_value'] = unique_rows.apply(max_utility, axis = 1)
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
    df_utilities_2016["max_value"] = df_with_calculated["max_value"]
    df_utilities_2016["best_option"] = df_with_calculated["best_option"]

    # best_options = df_utilities_2016.apply(lambda x: possible_outcomes[np.argmax(x)], axis=1)
    counts = df_utilities_2016["best_option"].value_counts()
    counts = counts.reindex(possible_outcomes, fill_value=0)

    # best_options_df = pd.DataFrame(best_options, columns = ["Best_option"])

    df_test = df_test.merge(df_utilities_2016, left_index=True, right_index=True)

    return df_test, counts, possible_outcomes



def reorder_df_with_limits(df, limits): 
    # Keep track of rows that have been processed
    sorted_rows = pd.DataFrame()
    processed_rows = pd.Series([False] * len(df))
    new_limits = limits.copy()

    selection_count = {col: 0 for col in df.columns}

    while True:
        

        # Identify valid columns that have not reached their selection limit
        valid_columns = [col for col in df.columns if selection_count[col] < limits[col]]
        if not valid_columns:
            break  # Stop if no valid columns are left

        temp_df = df[~processed_rows].copy()

        # Calculate max value for each row, considering only valid columns
        temp_df['max_value_w_lim'] = temp_df[valid_columns].max(axis=1)
        temp_df['best_option_w_lim'] = temp_df[valid_columns].idxmax(axis=1)

        lims_valid_columns = [new_limits[col] for col in valid_columns]
        batch_size = min( max(min(lims_valid_columns), 1), len(temp_df))

        # Sort rows by the max value, excluding already processed rows
        df_sorted = temp_df.sort_values(by='max_value_w_lim', ascending=False).head(batch_size)

        # Update selection count for each location in the batch
        for loc in df_sorted['best_option_w_lim']:
            selection_count[loc] += 1

        # Mark processed rows as True
        processed_rows[df_sorted.index] = True

        # Add the sorted rows to the result DataFrame
        sorted_rows = pd.concat([sorted_rows, df_sorted])

        # Print the selected rows and locations for this iteration (optional)
        # logger.info(f"Selected locations in this iteration:\n{df_sorted[['max_location']]}")
        # logger.info(df_sorted.drop(columns=['max_value', 'max_location']))
        
        new_limits = {key: (limits[key] - selection_count[key]) if new_limits[key] > 0 else new_limits[key] for key in valid_columns }  

        # Check if any location has reached its limit and stop considering it
        '''for loc in df_sorted['max_location_w_lim']:
            if selection_count[loc] >= new_limits[loc]:
                logger.info(f"Location {loc} has reached its limit.")'''

        # Stop if all rows are processed or no valid columns are left
        if df_sorted.shape[0] == 0 or all(selection_count[col] >= limits[col] for col in selection_count):
            break

        # # Drop the temporary columns used for sorting
        # sorted_rows = sorted_rows.drop(columns=['max_value', 'max_location'])

    return sorted_rows



def new_screening_strategy(df_test, net, possible_outcomes, counts, limit, operational_limit = None, logger=None, verbose = False):
    t1 = time.time()

    n_screening_tests = len(possible_outcomes) - 1

    # Setting limits to the numer of tests performed. 
    # For now, if the number of tests recommended is higher than the available tests, the number of tests is set to the available tests,
    # which are applied to the patients with the highest utility. The rest of the patients are not screened.


    if limit:
        if verbose:
            logger.info("Limited number of tests will be performed.")
        
        df_test_ordered = reorder_df_with_limits(df_test[possible_outcomes].copy(), operational_limit)
        df_test = df_test.merge(df_test_ordered[["max_value_w_lim", "best_option_w_lim"]], left_index=True, right_index=True)

    else:
        if verbose:
            logger.info("No limit on the number of tests.")


    df_test["Colonoscopy"] = None

    positive_predictions_count = pd.DataFrame(index = possible_outcomes, columns = ["Positive_predictions_by_screening", "Positive_predictions_by_colonoscopy"])
    positive_predictions_counts = 0


    for i, strategy in enumerate(possible_outcomes):
        
        if strategy == "No_scr_no_col":
            if limit:
                selected_patients = df_test.loc[df_test[df_test["best_option_w_lim"] == possible_outcomes[i]].index].copy()
            else:
                selected_patients = df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index].copy()

            y_selected_patients = selected_patients["CRC"]

            df_test.loc[selected_patients.index, "Prediction_screening"] = 0
            df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = 0
            df_test.loc[selected_patients.index, "Final_decision"] = 0
            continue


        elif strategy == "No_scr_col":
            if limit: 
                selected_patients = df_test.loc[df_test[df_test["best_option_w_lim"] == possible_outcomes[i]].index].copy()
            else:
                selected_patients = df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index].copy()

            y_selected_patients = selected_patients["CRC"]

            df_test.loc[selected_patients.index, "Prediction_screening"] = 0

            spec_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[0, 1 ,1]
            sens_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[1, 1 ,2]

            df_col = simulate_test_results(sens_col, spec_col, y_selected_patients)

            df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = df_col["TestResult"]
            df_test.loc[selected_patients.index, "Final_decision"] = df_col["TestResult"]

            positive_predictions_count.loc[possible_outcomes[i], "Positive_predictions_by_colonoscopy"] = df_col["TestResult"].sum()
            continue


        else:
            if limit:
                selected_patients = df_test.loc[df_test[df_test["best_option_w_lim"] == possible_outcomes[i]].index].copy()
            else:
                selected_patients = df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index].copy()

            if verbose:
                logger.info(f"** Testing strategy: {strategy} **")
                n_recommended_tests = len(df_test.loc[df_test[df_test["best_option"] == possible_outcomes[i]].index])
                n_tests_performed = len(selected_patients)
                logger.info(f"--> Number of recommended tests: {n_recommended_tests}")  
                logger.info(f"--> Number of tests performed: {n_tests_performed} \n")

            y_selected_patients = selected_patients["CRC"]

            spec_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,n_screening_tests,3)[0, i-1, 1]
            sens_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,n_screening_tests,3)[1, i-1, 2]

            df_scr = simulate_test_results(sens_strategy, spec_strategy, y_selected_patients) 

            df_test.loc[selected_patients.index, "Prediction_screening"] = df_scr["TestResult"]
            df_test.loc[selected_patients.index, "Prediction_colonoscopy"] = 0
            df_test.loc[selected_patients.index, "Final_decision"] = df_scr["TestResult"]

            # For SCR positives, perform colonoscopy:
            SCR_positives = df_scr[df_scr["TestResult"] == 1]
            y_SCR_positives = SCR_positives["Condition"]

            spec_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[0, 1 ,1]
            sens_col = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)[1, 1 ,2]
            
            df_col = simulate_test_results(sens_col, spec_col, y_SCR_positives)

            df_test.loc[SCR_positives.index, "Colonoscopy"] = "Colonoscopy"
            df_test.loc[SCR_positives.index, "Prediction_colonoscopy"] = df_col["TestResult"]
            df_test.loc[SCR_positives.index, "Final_decision"] = df_col["TestResult"]

            positive_predictions_count.loc[possible_outcomes[i], "Positive_predictions_by_screening"] = df_scr["TestResult"].sum()
            positive_predictions_count.loc[possible_outcomes[i], "Positive_predictions_by_colonoscopy"] = df_col["TestResult"].sum()


    net.update_beliefs()
    total_cost_of_screening = 0.0
    total_cost_of_colonoscopy = 0.0
    scr_costs = net.get_node_value("COST")[::2]
    scr_costs.insert(1, 0.0)

    col_costs = net.get_node_value("COST")[1]

    if limit:
        counts = df_test["best_option_w_lim"].value_counts()
    else: 
        counts = df_test["best_option"].value_counts()

    counts = counts.reindex(possible_outcomes, fill_value=0)

    total_cost_of_screening = np.matmul(np.array(scr_costs),counts.values)

    try:
        num_colonoscopies = df_test["Colonoscopy"].value_counts()["Colonoscopy"]
    except:
        num_colonoscopies = 0.0

    total_cost_of_colonoscopy = col_costs * num_colonoscopies

    if verbose:
        logger.info(f"Total number of colonoscopies performed: {num_colonoscopies}")


    total_cost = total_cost_of_screening + total_cost_of_colonoscopy
    t2 = time.time()
    time_taken = t2 - t1


    return df_test, total_cost, time_taken, positive_predictions_count



def old_screening_strategy(df_test, net, possible_outcomes, logger = None, verbose = False):

    n_screening_tests = len(possible_outcomes) - 1

    t1 = time.time()

    df_test["best_option"] = "No_scr_no_col"
    df_test["Prediction_screening"] = 0
    df_test["Colonoscopy"] = None
    df_test["Prediction_colonoscopy"] = 0
    df_test["Final_decision"] = 0
    df_test.loc[df_test["Age"] == "age_5_old_adult","best_option"] = "FIT"
    

    selected_patients = df_test[df_test["Age"] == "age_5_old_adult"].copy()
    y_selected_patients = selected_patients["CRC"]

    # Do FIT to all patients over 50
    spec_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,n_screening_tests,3)[0, 2, 1]
    sens_strategy = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,n_screening_tests,3)[1, 2, 2]

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

    if verbose:
        n_colonoscopies = df_test["Colonoscopy"].value_counts()["Colonoscopy"]
        logger.info(f"Total number of colonoscopies performed: {n_colonoscopies}")

    total_cost = total_cost_of_screening + total_cost_of_colonoscopy
    t2 = time.time()
    time_taken = t2 - t1

    return df_test, total_cost, time_taken



def compare_strategies(df_selected, net, possible_outcomes, operational_limit = None, logger=None, verbose = False):
    # A naive comparison would be to order the patients in terms of utility and perform recommended test + colonoscopy if needed 
    # in order until money runs out.
    logger.info("Naively comparing strategies with a max cost of ", operational_limit)

    t1 = time.time()

    df_temp = df_selected.sort_values(by="max_value", ascending=False).copy()
    df_temp["best_option_w_lim"] = "No_scr_no_col"
    df_temp["colonoscopy_w_lim"] = None
    cost = 0.0

    scr_costs = net.get_node_value("COST")[::2]
    scr_costs.insert(1, 0.0)
    col_costs = net.get_node_value("COST")[1]

    for i in df_temp.index:
        if cost > operational_limit:
            logger.info("Cost limit reached at patient number ", len(df_temp.loc[:i])) 
            break
        else:

            df_temp.loc[i, "best_option_w_lim"] = df_temp.loc[i, "best_option"]
            cost += scr_costs[possible_outcomes.index(df_temp.loc[i, "best_option"])]

            if df_temp.loc[i, "Prediction_screening"] == 1:
                df_temp.loc[i, "colonoscopy_w_lim"] = "Colonoscopy"
                cost += col_costs

    return df_temp, cost
            

            




# def simulate_test_results(sensitivity_scr, specificity_scr,
#                            sensitivity_col, specificity_col, y_crc):
#     """
#     Simulate test results based on sensitivity, specificity, and actual number of patients
#     with and without the disease.
    
#     Parameters:
#     - sensitivity (float): Sensitivity of the test (true positive rate)
#     - specificity (float): Specificity of the test (true negative rate)
#     - num_with_disease (int): Number of patients who have the disease
#     - num_without_disease (int): Number of patients who do not have the disease
    
#     Returns:
#     - pandas DataFrame: A DataFrame with the simulated test results, true conditions, and test outcomes.
#     """
    

#     num_with_disease = y_crc.sum()
#     num_without_disease = len(y_crc) - num_with_disease

#     # Step 1: Create a list of patients with and without the disease
    
#     # Step 2: Simulate test results
#     scr_results = []
#     col_results = []
    
#     for y in y_crc:
#         if y == 1:
#             # Patient has the disease, test is positive with probability = sensitivity
#             scr_result = np.random.choice([1, 0], p=[sensitivity_scr, 1 - sensitivity_scr])
#         else:
#             # Patient does not have the disease, test is negative with probability = specificity
#             scr_result = np.random.choice([0, 1], p=[specificity_scr, 1 - specificity_scr])

#         scr_results.append(scr_result)
    
#     # Step 3: Create a DataFrame to store the results
#     df_scr = pd.DataFrame({
#         'Condition': y_crc,    # True condition of the patient
#         'TestResult': scr_results  # Simulated test result
#     })


#     # For FIT positives, perform colonoscopy:
#     FIT_positives = df_scr[df_scr["TestResult"] == 1]

#     conditions = FIT_positives["Condition"].to_list()

#     col_results = []

#     for condition in conditions:
#         if condition == 1:
#             # Patient has the disease, test is positive with probability = sensitivity
#             col_result = np.random.choice([1, 0], p=[sensitivity_col, 1 - sensitivity_col])
#         else:
#             # Patient does not have the disease, test is negative with probability = specificity
#             col_result = np.random.choice([0, 1], p=[specificity_col, 1 - specificity_col])

#         col_results.append(col_result)

#     # Step 5: Create a DataFrame to store the results
#     df_col = pd.DataFrame({
#         'Condition': conditions,    # True condition of the patient
#         'TestResult': col_results  # Simulated test result
#     })

    
    
#     return df_scr, df_col



# def output_test_results(df_test, y,df_scr, cost_scr, df_col, cost_col, verbose = False):

#     # Add columns to indicate true positives, false positives, etc.
#     df_scr['TruePositive'] = (df_scr['Condition'] == 1) & (df_scr['TestResult'] == 1)
#     df_scr['FalsePositive'] = (df_scr['Condition'] == 0) & (df_scr['TestResult'] == 1)
#     df_scr['TrueNegative'] = (df_scr['Condition'] == 0) & (df_scr['TestResult'] == 0)
#     df_scr['FalseNegative'] = (df_scr['Condition'] == 1) & (df_scr['TestResult'] == 0)
    
#     # Step 4: Calculate confusion matrix components
#     TP_scr = df_scr['TruePositive'].sum()
#     FP_scr = df_scr['FalsePositive'].sum()
#     TN_scr = df_scr['TrueNegative'].sum()
#     FN_scr = df_scr['FalseNegative'].sum()
    
#     # Create confusion matrix
#     confusion_matrix_scr = pd.DataFrame({
#         'Predicted Negative': [TN_scr, FN_scr],
#         'Predicted Positive': [FP_scr, TP_scr]
#     }, index=['Actual Negative', 'Actual Positive'])


#     FIT_positives = df_scr[df_scr["TestResult"] == 1]
#     patient_data = df_scr["Condition"]

#     if verbose:
#         logger.info("Number of patients considered: ", patient_data.shape[0])
#         logger.info(f"Cost of screening: {cost_scr*(patient_data.shape[0])} €")
#         logger.info("Number of FIT positives: ", FIT_positives.shape[0])
#         logger.info("Number of colonoscopies to be done: ", FIT_positives.shape[0])
#         logger.info(f"Cost of colonoscopy program: {cost_col*FIT_positives.shape[0]} €")

    
#     # Add columns to indicate true positives, false positives, etc.
#     df_col['TruePositive'] = (df_col['Condition'] == 1) & (df_col['TestResult'] == 1)
#     df_col['FalsePositive'] = (df_col['Condition'] == 0) & (df_col['TestResult'] == 1)
#     df_col['TrueNegative'] = (df_col['Condition'] == 0) & (df_col['TestResult'] == 0)
#     df_col['FalseNegative'] = (df_col['Condition'] == 1) & (df_col['TestResult'] == 0)

#     # Step 6: Calculate confusion matrix components
#     TP_col = df_col['TruePositive'].sum()
#     FP_col = df_col['FalsePositive'].sum()
#     TN_col = df_col['TrueNegative'].sum()
#     FN_col = df_col['FalseNegative'].sum()

#     # Create confusion matrix
#     confusion_matrix_col = pd.DataFrame({
#         'Predicted Negative': [TN_col, FN_col],
#         'Predicted Positive': [FP_col, TP_col]
#     }, index=['Actual Negative', 'Actual Positive'])

#     total_cost = cost_scr*df_scr["Condition"].shape[0] + cost_col*FIT_positives.shape[0]

#     if verbose:
#         logger.info("Number of CRC true positive cases detected by colonoscopy: ", TP_scr)
#         logger.info("Number of false positives by colonoscopy: ", FP_scr)
#         logger.info(f"Total cost of screening and colonoscopy: {total_cost} €")
#         logger.info("Proportion of total CRC cases in the whole population detected by the method: ", TP_scr / df_test["CRC"].sum())
#         logger.info("Proportion of cases in the high-risk target population detected by the method: ", TP_scr / y.sum())

#     combined_confusion_matrix = pd.DataFrame({
#         'Predicted Negative': [TN_scr + TN_col, FN_scr + FN_col],
#         'Predicted Positive': [FP_col, TP_col]
#     }, index=['Actual Negative', 'Actual Positive'])

#     # Calculate sensitivity and specificity using the combined confusion matrix
#     sensitivity = TP_col / (TP_col + FN_col + FN_scr)
#     specificity = (TN_scr + TN_col) / (TN_scr +TN_col + FP_col)
#     PPV = TP_col / (TP_col + FP_col)
#     NPV = (TN_scr + TN_col) / (TN_scr + TN_col + FN_scr + FN_col)

#     metrics = {
#         "sensitivity": sensitivity,
#         "specificity": specificity,
#         "PPV": PPV,
#         "NPV": NPV
#     }
    
#     return confusion_matrix_scr, confusion_matrix_col, combined_confusion_matrix, total_cost, metrics
    


def create_folders_logger(single_run, label):
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, 'logs')
    # Create the logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join(log_dir, date_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    time_str = datetime.datetime.now().strftime("%H-%M-%S")
    log_dir = os.path.join(log_dir, label + time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{label}{timestamp}_singlerun_{single_run}.log")

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the minimum logging level

    # Create handlers for file and console
    file_handler = logging.FileHandler(log_filename)  # Logs to file
    console_handler = logging.StreamHandler()  # Logs to console

    # Set the logging level for both handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Define the formatter for logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_dir
