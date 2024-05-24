import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import conditional_mutual_info, pointwise_conditional_mutual_info, cond_kl_divergence

np.seterr(divide='ignore', invalid = 'ignore')

import matplotlib.pyplot as plt

# Make an array and iterate over possible values of probabilities
def plot_cond_mut_info(net):

    arr = []
    i = 0

    for prob in np.arange(0, 1.01, 0.01):

        p_CRC_false, p_CRC_true = [1-prob, prob] 

        p_y = np.array([p_CRC_false, p_CRC_true])
        H_y = np.sum(p_y * np.log(1 / p_y) )
        H_y

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

        aux_arr_scr = df_plotted_scr.sum(axis = 0).values.reshape(2,7).sum(axis = 0)

    

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

        aux_arr_col = df_plotted_col.sum(axis = 0).values.reshape(2,2).sum(axis = 0)

        arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)  

        # arr = np.append(arr, df_plotted_scr.sum(axis = 0).values.reshape(2,7).sum(axis = 0), 0)
    
    arr = arr.reshape(101,8)
    arr = arr.transpose()


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")
        leg = plt.legend(loc='upper right', shadow=True)
        title = "Conditional Mutual Information for Screening"
        plt.title(title)

    # save plot
    plt.savefig(f"output_images/{title}.png")

    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")
        leg = plt.legend(loc='upper right', shadow=True)
        
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0,0.1)

    plt.savefig(f"output_images/{title}_zoom.png")


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")
        leg = plt.legend(loc='upper right', shadow=True)
        
    ax.set_xlim(0.90, 1)
    ax.set_ylim(0,0.1)

    plt.savefig(f"output_images/{title}_zoom2.png")

    return