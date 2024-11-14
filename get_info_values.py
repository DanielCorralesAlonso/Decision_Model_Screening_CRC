import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
import pdb

np.seterr(divide='ignore', invalid = 'ignore', over = 'ignore')


def mutual_info_measures(net, plot = False, p_CRC_false = None, p_CRC_true = None, normalize = False, weighted = False):

    n = net.get_outcome_count("Screening")

    try:
        net.delete_arc("Results_of_Screening", "Colonoscopy")
        net.update_beliefs()
    except:
        pass
    
    if not plot:
        p_CRC_false, p_CRC_true = net.get_node_value("CRC")
    else:
        pass

    # --- Screening -----------------------------------------------------------
    point_cond_mut_info_scr, cond_mut_info_scr = calculate_values(net, p_CRC_false, p_CRC_true, "Screening", "Results_of_Screening")
    
    p_y = np.array([p_CRC_false, p_CRC_true])
    H_y = np.sum(p_y * np.log(1 / p_y) )

    rel_point_cond_mut_info_scr = point_cond_mut_info_scr / H_y
    rel_point_cond_mut_info_scr = np.nan_to_num(rel_point_cond_mut_info_scr, 0)

    rel_cond_mut_info_scr = cond_mut_info_scr / H_y
    rel_cond_mut_info_scr = np.nan_to_num(rel_cond_mut_info_scr, 0)
    
    df_plotted_scr = plot_df(point_cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])

    # --- Colonoscopy ---------------------------------------------------------

    net.update_beliefs()
    if not plot:
        try:
            net.add_arc("Results_of_Screening", "Colonoscopy")
            net.update_beliefs()
        except:
            pass

    point_cond_mut_info_col_array = []
    rel_point_cond_mut_info_col_array = []
    cond_mut_info_col_array = []
    rel_cond_mut_info_col_array = []

    if plot:
        # For the plot of the RMI of colonoscopy, we do not consider the screening node
        
        point_cond_mut_info_col, cond_mut_info_col = calculate_values(net, p_CRC_false, p_CRC_true, "Colonoscopy", "Results_of_Colonoscopy")
        
        p_y = np.array([p_CRC_false, p_CRC_true])
        H_y = np.sum(p_y * np.log(1 / p_y) )

        rel_point_cond_mut_info_col = point_cond_mut_info_col / H_y
        rel_point_cond_mut_info_col = np.nan_to_num(rel_point_cond_mut_info_col, 0)

        rel_cond_mut_info_col = cond_mut_info_col / H_y
        rel_cond_mut_info_col = np.nan_to_num(rel_cond_mut_info_col, 0)
        
        df_plotted_col = plot_df(point_cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])

        dict_scr = {"point_cond_mut_info": point_cond_mut_info_scr, "rel_point_cond_mut_info": rel_point_cond_mut_info_scr, "cond_mut_info": cond_mut_info_scr, "rel_cond_mut_info": rel_cond_mut_info_scr}
        dict_col = {"point_cond_mut_info": point_cond_mut_info_col, "rel_point_cond_mut_info": rel_point_cond_mut_info_col, "cond_mut_info": cond_mut_info_col, "rel_cond_mut_info": rel_cond_mut_info_col}

        dict = {}


    else:

        net.set_evidence("Screening", "No_screening")
        net.set_evidence("Results_of_Screening", "No_pred_screening")
        net.update_beliefs()

        p_CRC_false_prior, p_CRC_true_prior = net.get_node_value("CRC")
        for scr in net.get_outcome_ids("Screening"):
            net.set_evidence("Screening", scr)

            for elem in net.get_outcome_ids("Results_of_Screening"):
                net.update_beliefs()
                try: 
                    net.set_evidence("Results_of_Screening", elem)
                    
                    net.update_beliefs()

                    p_CRC_false_pos, p_CRC_true_pos = net.get_node_value("CRC")
                    
                    point_cond_mut_info_col, cond_mut_info_col = calculate_values(net, p_CRC_false_pos, p_CRC_true_pos, "Colonoscopy", "Results_of_Colonoscopy")
                    
                    p_y = np.array([p_CRC_false_prior, p_CRC_true_prior])
                    H_y = np.sum(p_y * np.log(1 / p_y) )

                    rel_point_cond_mut_info_col = point_cond_mut_info_col / H_y
                    rel_point_cond_mut_info_col = np.nan_to_num(rel_point_cond_mut_info_col, 0)

                    rel_cond_mut_info_col = cond_mut_info_col / H_y
                    rel_cond_mut_info_col = np.nan_to_num(rel_cond_mut_info_col, 0)

                    df_plotted_col = plot_df(point_cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])
                
                except:
                    point_cond_mut_info_col = np.zeros((2,2,3))
                    rel_point_cond_mut_info_col = np.zeros((2,2,3))
                    cond_mut_info_col = np.zeros((2,2,3))
                    rel_cond_mut_info_col = np.zeros((2,2,3))

                point_cond_mut_info_col_array.append(point_cond_mut_info_col)
                rel_point_cond_mut_info_col_array.append(rel_point_cond_mut_info_col)
                cond_mut_info_col_array.append(cond_mut_info_col)
                rel_cond_mut_info_col_array.append(rel_cond_mut_info_col)

            
                net.clear_evidence("Results_of_Screening")

            net.clear_evidence("Screening")
            net.update_beliefs()
        
        point_cond_mut_info_col_array = np.stack(point_cond_mut_info_col_array, axis = 0).reshape(n,3,2,2,3)
        rel_point_cond_mut_info_col_array = np.stack(rel_point_cond_mut_info_col_array, axis = 0).reshape(n,3,2,2,3)
        cond_mut_info_col_array = np.stack(cond_mut_info_col_array, axis = 0).reshape(n,3,2,2,3)
        rel_cond_mut_info_col_array = np.stack(rel_cond_mut_info_col_array, axis = 0).reshape(n,3,2,2,3)

        point_cond_mut_info = point_cond_mut_info_col_array
        rel_point_cond_mut_info = rel_point_cond_mut_info_col_array
        cond_mut_info = cond_mut_info_col_array
        rel_cond_mut_info = rel_cond_mut_info_col_array

        for i_scr in range(len(net.get_outcome_ids("Screening"))):
            for i_res_scr in range(len(net.get_outcome_ids("Results_of_Screening"))):
                # if not i_res_scr == 0:
                    point_cond_mut_info[i_scr, i_res_scr, :, 0, 0] = point_cond_mut_info_scr[:, i_scr, i_res_scr]
                    point_cond_mut_info[i_scr, i_res_scr, :, 1, 1:] = point_cond_mut_info_col_array[i_scr, i_res_scr, :, 1, 1:] +  point_cond_mut_info_scr[:, i_scr, i_res_scr].reshape(1,-1).transpose()

                    rel_point_cond_mut_info[i_scr, i_res_scr, :, 0, 0] = rel_point_cond_mut_info_scr[:, i_scr, i_res_scr]
                    rel_point_cond_mut_info[i_scr, i_res_scr, :, 1, 1:] = rel_point_cond_mut_info_col_array[i_scr, i_res_scr, :, 1, 1:] +  rel_point_cond_mut_info_scr[:, i_scr, i_res_scr].reshape(1,-1).transpose()

                    cond_mut_info[i_scr, i_res_scr, :, 0, 0] = cond_mut_info_scr[:, i_scr, i_res_scr]
                    cond_mut_info[i_scr, i_res_scr, :, 1, 1:] = cond_mut_info_col_array[i_scr, i_res_scr, :, 1, 1:] +  cond_mut_info_scr[:, i_scr, i_res_scr].reshape(1,-1).transpose()

                    rel_cond_mut_info[i_scr, i_res_scr, :, 0, 0] = rel_cond_mut_info_scr[:, i_scr, i_res_scr]
                    rel_cond_mut_info[i_scr, i_res_scr, :, 1, 1:] = rel_cond_mut_info_col_array[i_scr, i_res_scr, :, 1, 1:] +  rel_cond_mut_info_scr[:, i_scr, i_res_scr].reshape(1,-1).transpose()


        # pdb.set_trace() 

        dict_scr = {"point_cond_mut_info": point_cond_mut_info_scr, "rel_point_cond_mut_info": rel_point_cond_mut_info_scr, "cond_mut_info": cond_mut_info_scr, "rel_cond_mut_info": rel_cond_mut_info_scr}
        dict_col = {"point_cond_mut_info": point_cond_mut_info_col_array, "rel_point_cond_mut_info": rel_point_cond_mut_info_col_array, "cond_mut_info": cond_mut_info_col_array, "rel_cond_mut_info": rel_cond_mut_info_col_array}

        dict = {"point_cond_mut_info": point_cond_mut_info, "rel_point_cond_mut_info": rel_point_cond_mut_info, "cond_mut_info": cond_mut_info, "rel_cond_mut_info": rel_cond_mut_info}


    return dict, dict_scr, dict_col


# def calculate_values_new(net, scr, res_scr, p_CRC_false, p_CRC_true, decision_node, chance_node, normalize = False, weighted = False):



def calculate_values(net, p_CRC_false, p_CRC_true, decision_node, value_node, normalize = False, weighted = False):

    p_y = np.array([p_CRC_false, p_CRC_true])
    H_y = np.sum(p_y * np.log(1 / p_y) )

    n = net.get_outcome_count(decision_node)

    p_x_yz = np.array(net.get_node_definition(value_node)).reshape(2,n,3)
    
    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 3*n).reshape(2,n,3)

    p_x_z = p_y * p_x_yz
    p_x_z = np.sum(p_x_z, axis = 0)
    p_x_z = np.tile(p_x_z, (2,1)).reshape((2,n,3))

    if weighted:
        point_cond_mut_info = np.log( p_x_yz.reshape((2,n,3)) / p_x_z ) * ((1-p_y))
        point_cond_mut_info = np.nan_to_num(point_cond_mut_info, 0)
    else:
        point_cond_mut_info = np.log( p_x_yz.reshape((2,n,3)) / p_x_z )
        point_cond_mut_info = np.nan_to_num(point_cond_mut_info, 0)


    cond_mut_info = (p_y * ( p_x_yz * point_cond_mut_info ) )# .reshape(2,n,3))
    cond_mut_info = np.nan_to_num(cond_mut_info, 0)

    '''rel_point_cond_mut_info = point_cond_mut_info / H_y
    rel_point_cond_mut_info = np.nan_to_num(rel_point_cond_mut_info, 0)

    rel_cond_mut_info = cond_mut_info / H_y
    rel_cond_mut_info = np.nan_to_num(rel_cond_mut_info, 0)'''


    return point_cond_mut_info, cond_mut_info




def cond_kl_divergence(net):
    net.set_evidence("CRC", False)
    p_CRC_false = net.prob_evidence()

    net.set_evidence("CRC", True)
    p_CRC_true = net.prob_evidence()

    # --- Screening -----------------------------------------------------------

    p_x_yz = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)
    
    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 21).reshape(2,7,3)

    p_xy_z = p_y*(p_x_yz)

    p_x_z = p_y * p_x_yz
    p_x_z = np.sum(p_x_z, axis = 0)
    p_x_z = np.tile(p_x_z, (2,1)).reshape((2,7,3))

    p_y_x = p_xy_z / p_x_z

    values_KL = p_y_x * np.log(p_y_x / p_y)
    values_KL = np.nan_to_num(values_KL, 0)

    pd.DataFrame(values_KL.flatten()).transpose().to_csv("value_of_info_csv/cond_kl_div_scr.csv")

    df_plotted_scr = plot_df(values_KL, net, ["Results_of_Screening", "CRC", "Screening"])

    # --- Colonoscopy ---------------------------------------------------------

    p_t_yc = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)
    p_t_yc = np.swapaxes(p_t_yc,0,1)

    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 6).reshape(2,2,3)

    p_ty_c = p_y * p_t_yc

    p_t_c = p_y * p_t_yc
    p_t_c = np.sum(p_t_c, axis = 0)
    p_t_c = np.tile(p_t_c, (2,1)).reshape((2,2,3))

    p_y_t = p_ty_c / p_t_c

    values_KL = p_y_t * np.log(p_y_t / p_y)
    values_KL = np.nan_to_num(values_KL, 0)

    pd.DataFrame(values_KL.flatten()).transpose().to_csv("value_of_info_csv/cond_kl_div_col.csv")

    df_plotted_col = plot_df(values_KL, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])

    return df_plotted_scr, df_plotted_col




