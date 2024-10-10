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
def plot_cond_mut_info(net1, net2 = None, subtitle = '', plot = True, zoom = (0.1, 0.1), step = 0.001, output_dir = None):

    net_array = [net1]
    num_scr = len(net1.get_outcome_ids("Screening"))
    if net2 is not None:
        net_array.append(net2)

    dict_net = {}
    for net in net_array:
        arr = []
        h_y_arr = []
        i = 0

        for prob in np.arange(0, 1+step, step):

            p_CRC_false, p_CRC_true = [1-prob, prob] 

            p_y = np.array([p_CRC_false, p_CRC_true])
            H_y = np.sum(p_y * np.log(1 / p_y) )

            dict_scr, dict_col = mutual_info_measures(net, p_CRC_false, p_CRC_true)

            cond_mut_info_scr = dict_scr["cond_mut_info"]
            cond_mut_info_col = dict_col["cond_mut_info"]

            #print("Relative Mutual Information for Screening:")
            aux_arr_scr = cond_mut_info_scr.sum(axis = 0).sum(axis = 1)
            #print(aux_arr_scr)

            #print("Relative Mutual Information for Colonoscopy:")
            aux_arr_col = cond_mut_info_col.sum(axis = 0).sum(axis = 1)
            #print(aux_arr_col)


            arr = np.append(arr, np.append(aux_arr_scr, np.expand_dims(aux_arr_col[1], axis = 0) ,0) , 0)   
            h_y_arr = np.append(h_y_arr, H_y)


        arr = arr.reshape(int(1 / step + 1),num_scr+1)
        arr = arr.transpose()

        h_y_arr = np.nan_to_num(h_y_arr, 0)

        dict_net[net] = [arr, h_y_arr]
   
        # Save the conditional mutual information for screening
        if subtitle == 'new_test':
            pd.DataFrame(arr).to_csv(f"{output_dir}/output_data/cond_mut_info_new_test.csv")
        else:
            pd.DataFrame(arr).to_csv(f"{output_dir}/output_data/cond_mut_info.csv")


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]

    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict[labels[screening]])
            ax.plot(x, y2, color = color_dict[labels[screening]])
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])    

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])

    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)

    title = "Conditional Mutual Information "
    plt.title(title)
    ax.set_xlabel("p(CRC)")
    ax.set_ylabel("CMI")
  
    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_bounds.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}.png", bbox_inches='tight')
    plt.close()



    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict[labels[screening]])
            ax.plot(x, y2, color = color_dict[labels[screening]])
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])
        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])   

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"]) 

    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
    title = "Conditional Mutual Information"
    plt.title(title)
     
    ax.set_xlim(0, zoom[0])
    ax.set_ylim(-5*step,zoom[1])

    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_bounds.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict[labels[screening]])
            ax.plot(x, y2, color = color_dict[labels[screening]])
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])

    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict["H(CRC)"])    

    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
    title = "Conditional Mutual Information"
    plt.title(title)
     
    ax.set_xlim(1 - zoom[0], 1)
    ax.set_ylim(-5*step,zoom[1])

    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_bounds.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}.png", bbox_inches='tight')
    plt.close()


    return





def plot_relative_cond_mut_info(net1, net2 = None, subtitle = '', zoom=(0.001, 0.1), step = 0.001, output_dir = None):
    net_array = [net1]
    num_scr = len(net1.get_outcome_ids("Screening"))
    if net2 is not None:
        net_array.append(net2)

    dict_net = {}
    for net in net_array:
        arr = []
        h_y_arr = []
        i = 0

        for prob in np.arange(0, 1+step, step):

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


        arr = arr.reshape(int(1 / step + 1),num_scr+1)
        arr = arr.transpose()

        h_y_arr = np.nan_to_num(h_y_arr, 0)

        dict_net[net] = [arr, h_y_arr]
   
        # Save the relative conditional mutual information for screening
    
        if subtitle == 'new_test':
            pd.DataFrame(arr).to_csv(f"{output_dir}/output_data/rel_cond_mut_info_new_test.csv")
        else:
            pd.DataFrame(arr).to_csv(f"{output_dir}/output_data/rel_cond_mut_info.csv")


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]

    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict[labels[screening]])
            ax.plot(x, y2, color = color_dict[labels[screening]])
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])    


    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)

    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
    ax.set_xlabel("p(CRC)")
    ax.set_ylabel("RCMI")
  
    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/rel_cond_mut_info_{subtitle}_bounds.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/rel_cond_mut_info_{subtitle}.png", bbox_inches='tight')
    plt.close()



    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict[labels[screening]])
            ax.plot(x, y2, color = color_dict[labels[screening]])
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])    

    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
     
    ax.set_xlim(0, zoom[0])
    ax.set_ylim(-5*step,zoom[1])

    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/rel_cond_mut_info_zoom_{subtitle}_bounds.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/rel_cond_mut_info_zoom_{subtitle}.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots()
    labels = net.get_outcome_ids("Screening") + ["Colonoscopy"]
    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict[labels[screening]])
            ax.plot(x, y2, color = color_dict[labels[screening]])
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict[labels[screening]])
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict[labels[screening]])    

    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
     
    ax.set_xlim(1 - zoom[0], 1)
    ax.set_ylim(-5*step,zoom[1])

    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/rel_cond_mut_info_zoom_2_{subtitle}_bounds.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/rel_cond_mut_info_zoom_2_{subtitle}.png", bbox_inches='tight')
    plt.close()


    return


def plot_estimations_w_error_bars(mean_report, std_report, SE_report, label = ""):
    # Flatten the dataframes to get 1D arrays for mean and std
    mean_flat = mean_report.values.flatten()
    std_flat = std_report.values.flatten()
    SE_flat = SE_report.values.flatten()

    # Get corresponding row and column index for each point
    rows, cols = np.indices(mean_report.shape)

    # Flatten the row and column indices
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()

    # Plot the mean values with error bars for the std
    fig, ax = plt.subplots()

    # Scatter plot with error bars
    ax.errorbar(cols_flat, rows_flat, yerr=std_flat, fmt='o', color='b', ecolor='r', capsize=5)

    for i, (mean, se) in enumerate(zip(mean_flat, std_flat)):
        annotation_text = f"{mean:.2f} ± {se:.2f}"
        ax.text(cols_flat[i], rows_flat[i], annotation_text, ha='left', va='bottom', fontsize=9, color='black')


    # Set ticks and labels
    ax.set_xticks(range(mean_report.shape[1]))
    ax.set_xticklabels(mean_report.columns)
    ax.set_yticks(range(mean_report.shape[0]))
    ax.set_yticklabels(mean_report.index)

    # Set labels
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.title('Mean (+/- std) at Each Cell')
    plt.grid(True)

    plt.savefig(f"outputs/{label}_mean_SE_plot.png")

    plt.close()



def plot_screening_counts(counts, possible_outcomes):
    print("Number of tests performed")
    print(counts)
    plt.plot(counts)
    plt.xticks(range(len(possible_outcomes)), possible_outcomes, rotation = 45)
    plt.xlabel("Screening outcome")
    plt.savefig("outputs/screening_counts.png")
    plt.close()

    return