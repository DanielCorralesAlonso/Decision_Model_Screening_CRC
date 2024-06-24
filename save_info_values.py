
import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools

from get_info_values import mutual_info_measures

def save_info_values(net, value_function = "pcmi", normalize = False, weighted = False):

    # Get all combinations of possible parent states
    parents = net.get_parent_ids("CRC")
    parent_states = [net.get_outcome_ids(parent) for parent in parents]
    parent_combinations = list(itertools.product(*parent_states))


    value_scr_array = []
    value_col_array = []


    for elem in parent_combinations:
        # net.clear_all_evidence()

        for i in range(len(parents)):
            net.set_evidence(parents[i], elem[i])
        net.update_beliefs()

        p_CRC_false, p_CRC_true = net.get_node_value("CRC")

        dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true, normalize = normalize, weighted = weighted)
        
        if value_function == "pcmi":
            value_scr = dict_scr["point_cond_mut_info"]
            value_col = dict_col["point_cond_mut_info"]  
        elif value_function == "rel_pcmi":
            value_scr = dict_scr["rel_point_cond_mut_info"]
            value_col = dict_col["rel_point_cond_mut_info"]
        elif value_function == "cmi":
            value_scr = dict_scr["cond_mut_info"]
            value_col = dict_col["cond_mut_info"]

        value_scr_array = np.concatenate((value_scr_array, value_scr.flatten()), axis = 0)
        value_col_array = np.concatenate((value_col_array, value_col.flatten()), axis = 0)


    # Create a dataframe with the values of the conditional mutual information for screening
    added_variables_scr = ["CRC", "Screening", "Results_of_Screening"]
    total_variables_scr = list(parents + added_variables_scr)
    total_variables_scr_states = [net.get_outcome_ids(variable) for variable in total_variables_scr]
    total_combinations_scr = list(itertools.product(*total_variables_scr_states))

    index = pd.MultiIndex.from_tuples(total_combinations_scr, names=total_variables_scr)
        
    df_value_scr = pd.DataFrame(value_scr_array.reshape(1,-1), index=["Value"], columns=index)
    df_value_scr.to_csv("value_of_info_csv/point_cond_mut_info_scr.csv")



    # Create a dataframe with the values of the conditional mutual information for colonoscopy
    added_variables_col = ["CRC", "Colonoscopy", "Results_of_Colonoscopy"]
    total_variables_col = list(parents + added_variables_col)
    total_variables_col_states = [net.get_outcome_ids(variable) for variable in total_variables_col]
    total_combinations_col = list(itertools.product(*total_variables_col_states))

    index = pd.MultiIndex.from_tuples(total_combinations_col, names=total_variables_col)

    df_value_col = pd.DataFrame(value_col_array.reshape(1,-1), index=["Value"], columns=index)
    df_value_col.to_csv("value_of_info_csv/point_cond_mut_info_col.csv")

    return df_value_scr, df_value_col