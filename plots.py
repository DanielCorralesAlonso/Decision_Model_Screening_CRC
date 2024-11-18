import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures

import datetime

np.seterr(divide='ignore', invalid = 'ignore')

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

import matplotlib.pyplot as plt
import pdb

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

            dict, dict_scr, dict_col = mutual_info_measures(net,plot = True,  p_CRC_false = p_CRC_false, p_CRC_true = p_CRC_true )

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

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)

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

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)

    title = "Conditional Mutual Information"
    plt.title(title)
     
    ax.set_xlim(0, zoom[0])
    ax.set_ylim(-5*step,zoom[1])

    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_bounds_leftzoom.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_leftzoom.png", bbox_inches='tight')
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

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
    title = "Conditional Mutual Information"
    plt.title(title)
     
    ax.set_xlim(1 - zoom[0], 1)
    ax.set_ylim(-5*step,zoom[1])

    if net2 is not None:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_bounds_rightzoom.png", bbox_inches='tight')     
    else:
        plt.savefig(f"{output_dir}/output_images/cond_mut_info_{subtitle}_rightzoom.png", bbox_inches='tight')
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

            dict, dict_scr, dict_col = mutual_info_measures(net,plot = True,  p_CRC_false = p_CRC_false, p_CRC_true = p_CRC_true )

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

        h_y_arr = np.nan_to_num(h_y_arr, nan=0)

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

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
    '''ordered_functions = ['Colonoscopy', 'Colon_capsule', 'sDNA', 'FIT', 'CTC', 'Blood_based', 'gFOBT', 'No_screening']
    leg = plt.legend(ordered_functions,)'''

    title = "Relative Reduction of Uncertainty with respect to CRC"
    plt.title(title)
    ax.set_xlabel("p(CRC)")
    ax.set_ylabel("RMI")
  
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

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
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

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
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


def plot_estimations_w_error_bars(mean_report, std_report, SE_report = None, label = "", log_dir = None):
    # Flatten the dataframes to get 1D arrays for mean and std
    mean_flat = mean_report.values.flatten()
    std_flat = std_report.values.flatten()
    # SE_flat = SE_report.values.flatten()

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

    plt.savefig(f"{log_dir}/mean_std_{label}_plot.png")

    plt.close(fig)



def plot_screening_counts(counts,  possible_outcomes, operational_limit, counts_w_lim = None, log_dir = None, lambda_list = None, label = '_'):
    fig, ax = plt.subplots()

    # Loop through each bar to add the text annotations and apply conditional styling
    for i, outcome in enumerate(possible_outcomes):
        count = counts.iloc[i]
        limit = operational_limit[outcome]

        # Check if count exceeds the operational limit and apply conditional styling
        if count > limit:
            # Create a dashed bar for counts exceeding the limit
            bar1 = ax.bar(outcome, count, color='none', edgecolor='steelblue', linestyle='--', linewidth=2, hatch='//', align = 'center')
            # Add a colored bar for the operational limit
            bar2 = ax.bar(outcome, limit, color='red', alpha=1, align = 'center')
        else:
            # Create a colored bar for counts under the limit
            bar1 = ax.bar(outcome, count, color='steelblue', align = 'center')
            # Add a light colored bar for the operational limit
            bar2 = ax.bar(outcome, limit, color='red', alpha=0.15,   align = 'center')

        ax.text(bar1[0].get_x() + bar1[0].get_width()/2, count + 5000, str(int(count)), ha='center', color='black', fontsize=10)

        try:
            ax.text(bar2[0].get_x() + bar2[0].get_width()/2, -11000, str(int(limit)), ha='center', color='red', fontsize=10)
        except:
            pass

    if counts_w_lim is not None:
        for i, outcome in enumerate(possible_outcomes):
            if operational_limit[outcome] != counts_w_lim[outcome]:
                bar1 = ax.bar(outcome, counts_w_lim[outcome], color='steelblue',  alpha=0.15,  align = 'center')
                ax.text(bar1[0].get_x() + bar1[0].get_width()/2, counts_w_lim[outcome] + 5000, str(int(counts_w_lim[outcome])), ha='center', color='steelblue', fontsize=10)


    ax.legend()

    ax.set_ylim(0, 355000)
    ax.set_xticks(range(len(possible_outcomes)), possible_outcomes, rotation = 45)
    ax.set_xlabel("Screening outcome")
    ax.set_ylabel("Number of tests")
    ax.set_title("Recommended Tests vs. Operational Limit")

    if lambda_list is not None:
        ax.text(0.7, 0.8, r"$\lambda_1 =" + f"{lambda_list[0]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)
        ax.text(0.7, 0.75, r"$\lambda_2 =" + f"{lambda_list[1]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)
        ax.text(0.7, 0.7, r"$\lambda_3 =" + f"{lambda_list[2]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)
        ax.text(0.7, 0.65, r"$\lambda_4 =" + f"{lambda_list[3]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)

    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    plt.savefig(f"{log_dir}/screening_counts_{label}_{timestamp}.png", bbox_inches='tight')
    plt.close(fig)

    return