import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures

np.seterr(divide='ignore', invalid = 'ignore')

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

import matplotlib.pyplot as plt

# Make an array and iterate over possible values of probabilities
def plot_cond_mut_info(net, subtitle = '', plot = True, zoom = (0.1, 0.1)):

    arr = []
    h_y_arr = []
    i = 0

    for prob in np.arange(0, 1.001, 0.001):

        p_CRC_false, p_CRC_true = [1-prob, prob] 

        p_y = np.array([p_CRC_false, p_CRC_true])
        H_y = np.sum(p_y * np.log(1 / p_y) )

        dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true)

        cond_mut_info_scr = dict_scr["cond_mut_info"]
        cond_mut_info_col = dict_col["cond_mut_info"]

        # Save the conditional mutual information for screening
        pd.DataFrame(cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_scr.csv")

        df_plotted_scr = plot_df(cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])

        num_scr = len(net.get_outcome_ids("Screening"))
        aux_arr_scr = df_plotted_scr.sum(axis = 0).values.reshape(2,num_scr).sum(axis = 0)


        # Save the conditional mutual information for colonoscopy
        pd.DataFrame((cond_mut_info_col).flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_col.csv")

        df_plotted_col = plot_df(cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])

        aux_arr_col = df_plotted_col.sum(axis = 0).values.reshape(2,2).sum(axis = 0)


        arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)  
        h_y_arr = np.append(h_y_arr, H_y)

    arr = arr.reshape(1001,num_scr+1)
    arr = arr.transpose()

    h_y_arr = np.nan_to_num(h_y_arr, 0)

    if plot:
        color_dict = cfg["colors"]

        fig, ax = plt.subplots()
        labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1.001,0.001), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])
        
        ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])
        leg = plt.legend(loc='upper right', shadow=True)
        title = "Conditional Mutual Information for Screening"
        plt.title(title)

        # save plot
        plt.savefig(f"output_images/{title}_{subtitle}.png")

        fig, ax = plt.subplots()
        labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1.001,0.001), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])
        
        ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])
        leg = plt.legend(loc='upper right', shadow=True)
            
        ax.set_xlim(0, zoom[0])
        ax.set_ylim(0, zoom[1])

        plt.savefig(f"output_images/{title}_{subtitle}_zoom.png")


        fig, ax = plt.subplots()
        labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
        
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1.001,0.001), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])
            
        ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])  
        leg = plt.legend(loc='upper right', shadow=True)
            
        ax.set_xlim(1 - zoom[0], 1)
        ax.set_ylim(-0.005, zoom[1])

        plt.savefig(f"output_images/{title}_{subtitle}_zoom2.png")

    return arr




def plot_cond_mut_info_bounds(net1, net2 = None):

    
    net_array = [net1]
    if net2 is not None:
        net_array.append(net2)

    dict_net = {}
    for net in net_array:
        arr = []
        h_y_arr = []
        i = 0

        for prob in np.arange(0, 1.001, 0.001):

            p_CRC_false, p_CRC_true = [1-prob, prob] 

            p_y = np.array([p_CRC_false, p_CRC_true])
            H_y = np.sum(p_y * np.log(1 / p_y) )

            dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true)

            cond_mut_info_scr = dict_scr["cond_mut_info"]
            cond_mut_info_col = dict_col["cond_mut_info"]
            

            # Save the conditional mutual information for screening
            pd.DataFrame(cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_scr.csv")

            df_plotted_scr = plot_df(cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])

            num_scr = len(net.get_outcome_ids("Screening"))
            aux_arr_scr = df_plotted_scr.sum(axis = 0).values.reshape(2,num_scr).sum(axis = 0)


            # Save the conditional mutual information for colonoscopy
            pd.DataFrame((cond_mut_info_col).flatten()).transpose().to_csv("value_of_info_csv/cond_mut_info_col.csv")

            df_plotted_col = plot_df(cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])

            aux_arr_col = df_plotted_col.sum(axis = 0).values.reshape(2,2).sum(axis = 0)


            arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)  
            h_y_arr = np.append(h_y_arr, H_y)

        arr = arr.reshape(1001,num_scr+1)
        arr = arr.transpose()

        h_y_arr = np.nan_to_num(h_y_arr, 0)

        dict_net[net] = [arr, h_y_arr]

    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]

    color_dict = cfg["colors"]

    for screening in range(arr.shape[0]):
        x = np.arange(0,1.001,0.001)
        y1 = dict_net[net_array[0]][0][screening]
        y2 = dict_net[net_array[1]][0][screening]

        ax.plot(x, y1, color = color_dict[labels[screening]])
        ax.plot(x, y2, color = color_dict[labels[screening]])
        ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])
    

    ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(CRC)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Conditional Mutual Information for Screening"
    plt.title(title)


    return





def plot_relative_cond_mut_info(net, subtitle = '', zoom=(0.001, 0.1)):

    arr = []
    # h_y_arr = []
    i = 0
    num_scr = len(net.get_outcome_ids("Screening"))
    color_dict = cfg["colors"]

    for prob in np.arange(0, 1.001, 0.001):

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
        # h_y_arr = np.append(h_y_arr, H_y)


    arr = arr.reshape(1001,num_scr+1)
    arr = arr.transpose()


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.001,0.001), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])

    # ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(Y)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
    ax.set_xlabel("p(CRC)")
    ax.set_ylabel("RCMI")

    plt.savefig(f"output_images/rel_cond_mut_info_{subtitle}.png")



    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    for screening in range(arr.shape[0]):
        ax.plot(np.arange(0,1.001,0.001), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])

    # ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(Y)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
     
    ax.set_xlim(0, zoom[0])
    ax.set_ylim(-0.005,zoom[1])

    plt.savefig(f"output_images/rel_cond_mut_info_zoom_{subtitle}.png")





def plot_relative_cond_mut_info_bounds(net1, net2 = None):
    net_array = [net1]
    if net2 is not None:
        net_array.append(net2)

    dict_net = {}
    for net in net_array:
        arr = []
        h_y_arr = []
        i = 0

        for prob in np.arange(0, 1.001, 0.001):

            p_CRC_false, p_CRC_true = [1-prob, prob] 

            p_y = np.array([p_CRC_false, p_CRC_true])
            H_y = np.sum(p_y * np.log(1 / p_y) )

            dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true)

            cond_mut_info_scr = dict_scr["rel_cond_mut_info"]
            cond_mut_info_col = dict_col["rel_cond_mut_info"]
            

            # Save the conditional mutual information for screening
            pd.DataFrame(cond_mut_info_scr.flatten()).transpose().to_csv("value_of_info_csv/rel_cond_mut_info_scr.csv")

            df_plotted_scr = plot_df(cond_mut_info_scr, net, ["Results_of_Screening", "CRC", "Screening"])

            num_scr = len(net.get_outcome_ids("Screening"))
            aux_arr_scr = df_plotted_scr.sum(axis = 0).values.reshape(2,num_scr).sum(axis = 0)


            # Save the conditional mutual information for colonoscopy
            pd.DataFrame((cond_mut_info_col).flatten()).transpose().to_csv("value_of_info_csv/rel_cond_mut_info_col.csv")

            df_plotted_col = plot_df(cond_mut_info_col, net, ["Results_of_Colonoscopy", "CRC", "Colonoscopy"])

            aux_arr_col = df_plotted_col.sum(axis = 0).values.reshape(2,2).sum(axis = 0)


            arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)  
            h_y_arr = np.append(h_y_arr, H_y)

        arr = arr.reshape(1001,num_scr+1)
        arr = arr.transpose()

        h_y_arr = np.nan_to_num(h_y_arr, 0)

        dict_net[net] = [arr, h_y_arr]

    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]

    color_dict = cfg["colors"]

    for screening in range(arr.shape[0]):
        x = np.arange(0,1.001,0.001)
        y1 = dict_net[net_array[0]][0][screening]
        y2 = dict_net[net_array[1]][0][screening]

        ax.plot(x, y1, color_dict[labels[screening]])
        ax.plot(x, y2, color = color_dict[labels[screening]])
        ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])
    

    # ax.plot(np.arange(0,1.001,0.001), h_y_arr, label = "H(CRC)")
    leg = plt.legend(loc='upper right', shadow=True)
    title = "Relative Conditional Mutual Information for Screening"
    plt.title(title)


    return