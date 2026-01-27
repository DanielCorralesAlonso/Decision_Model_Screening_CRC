
import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools

from get_info_values import mutual_info_measures

def save_info_values(net, value_function = "point_cond_mut_info", new_test = False, normalize = False, weighted = False, output_dir=""):

    # Get all combinations of possible parent states
    parents = net.get_parent_ids("CRC")
    parent_states = [net.get_outcome_ids(parent) for parent in parents]
    parent_combinations = list(itertools.product(*parent_states))

    value_scr_array = []
    value_col_array = []

    value_array = []

    for elem in parent_combinations:

        for i in range(len(parents)):
            net.set_evidence(parents[i], elem[i])

        net.update_beliefs() 
        dict, dict_scr, dict_col = mutual_info_measures(net, normalize = normalize, weighted = weighted)

        if value_function == "point_cond_mut_info":
            value_scr = dict_scr["point_cond_mut_info"]
            value_col = dict_col["point_cond_mut_info"]  

        elif value_function == "rel_point_cond_mut_info":
            value_scr = dict_scr["rel_point_cond_mut_info"]
            value_col = dict_col["rel_point_cond_mut_info"]

        elif value_function == "cond_mut_info":
            value_scr = dict_scr["cond_mut_info"]
            value_col = dict_col["cond_mut_info"]

        value_scr_array = np.concatenate((value_scr_array, value_scr.flatten()), axis = 0)
        value_col_array = np.concatenate((value_col_array, value_col.flatten()), axis = 0)

        value_array = np.concatenate((value_array, dict[value_function].flatten()), axis = 0)


    # Create a dataframe with the values of the conditional mutual information for colonoscopy
    added_variables = ["Screening", "Results_of_Screening", "CRC", "Colonoscopy", "Results_of_Colonoscopy"]
    total_variables = list(parents + added_variables)
    total_variables_states = [net.get_outcome_ids(variable) for variable in total_variables]
    total_combinations = list(itertools.product(*total_variables_states))

    index = pd.MultiIndex.from_tuples(total_combinations, names=total_variables)

    df_value = pd.DataFrame(value_array.reshape(1,-1), index=["Value"], columns=index)
    if new_test:
        df_value.to_csv(f"{output_dir}/output_data/{value_function}_new_test.csv")
    else:
        df_value.to_csv(f"{output_dir}/output_data/{value_function}.csv")

    return df_value