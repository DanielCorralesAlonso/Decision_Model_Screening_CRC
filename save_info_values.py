
import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools

from get_info_values import pointwise_conditional_mutual_info

def save_info_values(net, normalize = False, weighted = False):

    # Get all combinations of possible parent states
    parents = net.get_parent_ids("CRC")
    parent_states = [net.get_outcome_ids(parent) for parent in parents]
    parent_combinations = list(itertools.product(*parent_states))


    value_scr_array = []
    value_col_array = []


    for elem in parent_combinations:
        net.clear_all_evidence()

        for i in range(len(parents)):
            net.set_evidence(parents[i], elem[i])
        net.update_beliefs()

        p_CRC_false, p_CRC_true = net.get_node_value("CRC")

        point_cond_mut_info_scr, cond_mut_info_scr, point_cond_mut_info_col, cond_mut_info_col = pointwise_conditional_mutual_info(net, p_CRC_false, p_CRC_true, normalize = normalize, weighted = weighted)


        value_scr_array = np.concatenate((value_scr_array, point_cond_mut_info_scr.flatten()), axis = 0)
        value_col_array = np.concatenate((value_col_array, point_cond_mut_info_col.flatten()), axis = 0)


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