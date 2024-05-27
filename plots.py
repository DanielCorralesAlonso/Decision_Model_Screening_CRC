import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures, cond_kl_divergence, pcmi_cmi

np.seterr(divide='ignore', invalid = 'ignore')

import matplotlib.pyplot as plt

# Make an array and iterate over possible values of probabilities
def plot_cond_mut_info(net):

    arr = []
    h_y_arr = []
    i = 0

    for prob in np.arange(0, 1.01, 0.01):

        p_CRC_false, p_CRC_true = [1-prob, prob] 

        p_y = np.array([p_CRC_false, p_CRC_true])
        H_y = np.sum(p_y * np.log(1 / p_y) )

        dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true)

        cond_mut_info_scr = dict_scr["cond_mut_info"]
        cond_mut_info_col = dict_col["cond_mut_info"]
        

        # Save the conditional mutual information for screening
        pd.DataFrame(cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_scr.csv")

        df_plotted_scr = plot_df(cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])

        aux_arr_scr = df_plotted_scr.sum(axis = 0).values.reshape(2,7).sum(axis = 0)


        # Save the conditional mutual information for colonoscopy
        pd.DataFrame((cond_mut_info_col).flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_col.csv")

        df_plotted_col = plot_df(cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])

        aux_arr_col = df_plotted_col.sum(axis = 0).values.reshape(2,2).sum(axis = 0)


        arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)  
        h_y_arr = np.append(h_y_arr, H_y)

    arr = arr.reshape(101,8)
    arr = arr.transpose()

    h_y_arr = np.nan_to_num(h_y_arr, 0)


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")
    
    ax.plot(np.arange(0,1.01,0.01), h_y_arr, label = "H(Y)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Conditional Mutual Information for Screening"
    plt.title(title)

    # save plot
    plt.savefig(f"output_images/{title}.png")

    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")
    
    ax.plot(np.arange(0,1.01,0.01), h_y_arr, label = "H(Y)")
    leg = plt.legend(loc='upper right', shadow=True)
        
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0,0.1)

    plt.savefig(f"output_images/{title}_zoom.png")


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")
        
    ax.plot(np.arange(0,1.01,0.01), h_y_arr, label = "H(Y)")  
    leg = plt.legend(loc='upper right', shadow=True)
        
    ax.set_xlim(0.90, 1)
    ax.set_ylim(0,0.1)

    plt.savefig(f"output_images/{title}_zoom2.png")

    return



def plot_relative_cond_mut_info(net):

    arr = []
    i = 0

    for prob in np.arange(0, 1.01, 0.01):

        p_CRC_false, p_CRC_true = [1-prob, prob] 

        p_y = np.array([p_CRC_false, p_CRC_true])
        H_y = np.sum(p_y * np.log(1 / p_y) )

        dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true)

        rel_cond_mut_info_scr = dict_scr["rel_cond_mut_info"]
        rel_cond_mut_info_col = dict_col["rel_cond_mut_info"]

        #print("Relative Mutual Information for Screening:")
        aux_arr_scr = rel_cond_mut_info_scr.sum(axis = 0).sum(axis = 1)
        #print(aux_arr_scr)

        #print("Relative Mutual Information for Colonoscopy:")
        aux_arr_col = rel_cond_mut_info_col.sum(axis = 0).sum(axis = 1)
        #print(aux_arr_col)


        arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)   


    arr = arr.reshape(101,8)
    arr = arr.transpose()


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")

    # ax.plot(np.arange(0,1.01,0.01), h_y_arr, label = "H(Y)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)

    plt.savefig(f"output_images/rel_cond_mut_info.png")



    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.01,0.01), arr[screening], label = f"{labels[screening]}")

    # ax.plot(np.arange(0,1.01,0.01), h_y_arr, label = "H(Y)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
     
    ax.set_xlim(0, 0.01)
    ax.set_ylim(0,0.1)

    plt.savefig(f"output_images/rel_cond_mut_info_zoom.png")
