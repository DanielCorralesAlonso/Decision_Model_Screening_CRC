import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net


def conditional_mutual_info(net):
    net.set_evidence("CRC", False)
    p_CRC_false = net.prob_evidence()

    net.set_evidence("CRC", True)
    p_CRC_true = net.prob_evidence()

    p_y = np.array([p_CRC_false, p_CRC_true])
    H_y = np.sum(p_y * np.log(1 / p_y) )
    H_y

    # --- Screening -----------------------------------------------------------

    p_x_yz = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)

    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 21).reshape(2,7,3)

    p_x_z = p_y * p_x_yz
    p_x_z = np.sum(p_x_z, axis = 0)
    p_x_z = np.tile(p_x_z, (2,1)).reshape((2,7,3))

    p_x_yz.reshape((2,7,3))

    # Calculate the conditional entropy
    cond_mut_info_scr = (p_y * ( p_x_yz * np.log( p_x_yz.reshape((2,7,3)) / p_x_z ) ).reshape(2,7,3))
    cond_mut_info_scr = np.nan_to_num(cond_mut_info_scr, 0)

    # Save the conditional mutual information for screening
    pd.DataFrame(cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_scr.csv")

    df_plotted_scr = plot_df(cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])


    # --- Colonoscopy ---------------------------------------------------------

    # Get the entropy of the target variable given the evidence
    p_t_yc = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)
    p_t_yc = np.swapaxes(p_t_yc,0,1)

    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 6).reshape(2,2,3)

    p_t_c = p_y * p_t_yc
    p_t_c = np.sum(p_t_c, axis = 0)
    p_t_c = np.tile(p_t_c, (2,1)).reshape((2,2,3))

    p_t_yc.reshape((2,2,3))

    cond_mut_info_col = (p_y * (p_t_yc * np.log( p_t_yc.reshape((2,2,3)) / p_t_c ) ).reshape(2,2,3))
    cond_mut_info_col = np.nan_to_num(cond_mut_info_col, 0)

    pd.DataFrame((cond_mut_info_col).flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_col.csv")

    df_plotted_col = plot_df(cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])


    return df_plotted_scr, df_plotted_col



def pointwise_conditional_mutual_info(net, normalize = False):
    '''net.set_evidence("CRC", False)
    p_CRC_false = net.prob_evidence()

    net.set_evidence("CRC", True)
    p_CRC_true = net.prob_evidence()'''

    p_CRC_false, p_CRC_true = net.get_node_value("CRC")

    p_y = np.array([p_CRC_false, p_CRC_true])
    H_y = np.sum(p_y * np.log(1 / p_y) )
    H_y

    # --- Screening -----------------------------------------------------------

    p_x_yz = np.array(net.get_node_definition("Results_of_Screening")).reshape(2,7,3)
    
    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 21).reshape(2,7,3)

    p_x_z = p_y * p_x_yz
    p_x_z = np.sum(p_x_z, axis = 0)
    p_x_z = np.tile(p_x_z, (2,1)).reshape((2,7,3))

    if normalize:
        point_cond_mut_info_scr = np.log( p_x_yz.reshape((2,7,3)) / p_x_z ) /  - np.log( p_x_yz.reshape((2,7,3)) * p_y)
        point_cond_mut_info_scr = np.nan_to_num(point_cond_mut_info_scr, 0)
        # pd.DataFrame(point_cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/norm_point_cond_mut_info_scr.csv")
    else:
        point_cond_mut_info_scr = np.log( p_x_yz.reshape((2,7,3)) / p_x_z )
        point_cond_mut_info_scr = np.nan_to_num(point_cond_mut_info_scr, 0)
        # pd.DataFrame(point_cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/point_cond_mut_info_scr.csv")

    # Print the DataFrame
    df_plotted_scr = plot_df(point_cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])



    # --- Colonoscopy ---------------------------------------------------------

    p_t_yc = np.array(net.get_node_definition("Results_of_Colonoscopy")).reshape(2,2,3)
    p_t_yc = np.swapaxes(p_t_yc,0,1)

    p_y = np.array([p_CRC_false, p_CRC_true])
    p_y = np.repeat(p_y, 6).reshape(2,2,3)

    p_t_c = p_y * p_t_yc
    p_t_c = np.sum(p_t_c, axis = 0)
    p_t_c = np.tile(p_t_c, (2,1)).reshape((2,2,3))

    if normalize:
        point_cond_mut_info_col = np.log( p_t_yc.reshape((2,2,3)) / p_t_c ) / - np.log( p_t_yc.reshape((2,2,3)) * p_y)
        point_cond_mut_info_col = np.nan_to_num(point_cond_mut_info_col, 0)
        pd.DataFrame(point_cond_mut_info_col.flatten()).transpose().to_csv("value_of_info_csv/norm_point_cond_mut_info_col.csv")
    else:
        point_cond_mut_info_col = np.log( p_t_yc.reshape((2,2,3)) / p_t_c )
        point_cond_mut_info_col = np.nan_to_num(point_cond_mut_info_col, 0)
        pd.DataFrame(point_cond_mut_info_col.flatten()).transpose().to_csv("value_of_info_csv/point_cond_mut_info_col.csv")

    # Print the DataFrame
    df_plotted_col = plot_df(point_cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])



    return point_cond_mut_info_scr, df_plotted_scr, point_cond_mut_info_col, df_plotted_col


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




