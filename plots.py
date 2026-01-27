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
import matplotlib.colors as mcolors
import pdb

def assign_missing_colors(labels, color_dict):
    existing_colors = set(c.upper() for c in color_dict.values())
    cmap = plt.get_cmap('tab20') 
    idx = 0
    for label in labels:
        if label not in color_dict:
            while True:
                c_hex = mcolors.to_hex(cmap(idx % 20)).upper()
                idx += 1
                if c_hex not in existing_colors:
                    color_dict[label] = c_hex
                    existing_colors.add(c_hex)
                    break 
                if idx > 100:
                    color_dict[label] = c_hex
                    break
    return color_dict

# Make an array and iterate over possible values of probabilities
def plot_cond_mut_info(net1, net2 = None, subtitle = '', plot = True, zoom = (0.1, 0.1), step = 0.01, output_dir = None):

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

    assign_missing_colors(labels, cfg["colors"])
    color_dict = cfg["colors"]

    if net2 is not None:
        for screening in range(arr.shape[0]):
            x = np.arange(0,1+step,step)
            y1 = dict_net[net_array[0]][0][screening]
            y2 = dict_net[net_array[1]][0][screening]

            ax.plot(x, y1, color_dict.get(labels[screening], '#000000'))
            ax.plot(x, y2, color = color_dict.get(labels[screening], '#000000'))
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict.get(labels[screening], '#000000'))

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict.get("H(CRC)", 'gray'))
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict.get(labels[screening], '#000000'))    

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict.get("H(CRC)", 'gray'))

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

            ax.plot(x, y1, color_dict.get(labels[screening], '#000000'))
            ax.plot(x, y2, color = color_dict.get(labels[screening], '#000000'))
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict.get(labels[screening], '#000000'))
        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict.get("H(CRC)", 'gray'))
    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict.get(labels[screening], '#000000'))   

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict.get("H(CRC)", 'gray')) 

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

            ax.plot(x, y1, color_dict.get(labels[screening], '#000000'))
            ax.plot(x, y2, color = color_dict.get(labels[screening], '#000000'))
            ax.fill_between(x, y1, y2, alpha = 0.1, label = f"{labels[screening]}", color = color_dict.get(labels[screening], '#000000'))

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict.get("H(CRC)", 'gray'))

    else:
        for screening in range(arr.shape[0]):
            ax.plot(np.arange(0,1+step,step), arr[screening], label = f"{labels[screening]}", color = color_dict.get(labels[screening], '#000000'))

        ax.plot(np.arange(0,1+step,step), h_y_arr, label = "H(CRC)", color = color_dict.get("H(CRC)", 'gray'))    

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
    if net2 is not None:
        net_array.append(net2)

    # 1. OPTIMIZATION: Adaptive X-axis Steps
    # We need fine resolution only in [0, zoom[0]] and [1-zoom[0], 1]
    # For the middle part, we can use a coarser step.
    x_limit = zoom[0]
    fine_step = step
    coarse_step = max(step * 10, 0.01)

    # Ensure ranges don't overlap or cross weirdly if zoom is large
    if x_limit >= 0.5:
        # If zoom area covers half or more, just use fine step everywhere
        x_values = np.arange(0, 1 + fine_step, fine_step)
    else:
        x_left = np.arange(0, x_limit + fine_step, fine_step)
        # Ensure we start middle part aligned roughly
        x_right = np.arange(1 - x_limit, 1 + fine_step, fine_step)
        
        # Middle range
        mid_start = x_left[-1] + coarse_step
        mid_end = x_right[0] - coarse_step
        
        if mid_start < mid_end:
            x_mid = np.arange(mid_start, mid_end + coarse_step, coarse_step)
            x_values = np.unique(np.concatenate((x_left, x_mid, x_right)))
        else:
             x_values = np.unique(np.concatenate((x_left, x_right)))

    # ensure exactly [0, 1] bounds and sorted
    x_values = x_values[(x_values >= 0) & (x_values <= 1)]
    x_values.sort()

    dict_net = {}
    
    # 2. CALCULATION LOOP
    for net in net_array:
        list_data = [] # List to collect screening+colonoscopy vectors
        list_h_y = []

        for prob in x_values:
            p_CRC_false, p_CRC_true = [1-prob, prob] 

            # Entropy H(Y)
            p_y = np.array([p_CRC_false, p_CRC_true])
            # Handle log(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_p_y = np.log(1 / p_y)
                log_p_y[np.isinf(log_p_y)] = 0
                H_y = np.sum(p_y * log_p_y) 

            dict, dict_scr, dict_col = mutual_info_measures(net, plot = True,  p_CRC_false = p_CRC_false, p_CRC_true = p_CRC_true )

            rel_cond_mut_info_scr = dict_scr["rel_cond_mut_info"]
            rel_cond_mut_info_col = dict_col["rel_cond_mut_info"]

            # Summing
            aux_arr_scr = rel_cond_mut_info_scr.sum(axis = 0).sum(axis = 1)
            aux_arr_col = rel_cond_mut_info_col.sum(axis = 0).sum(axis = 1)
            
            # Combine: Screening vars + Colonoscopy (matches original structure)
            combined_row = np.concatenate((aux_arr_scr, [aux_arr_col[1]]))
            
            list_data.append(combined_row)
            list_h_y.append(H_y)

        # Structure: (num_vars, num_points)
        arr = np.array(list_data).T 
        h_y_arr = np.array(list_h_y)
        h_y_arr = np.nan_to_num(h_y_arr, nan=0)

        dict_net[net] = [arr, h_y_arr]
        
        # Save CSV (with index since steps are non-uniform)
        df_save = pd.DataFrame(arr.T)
        df_save.index = x_values
        df_save.index.name = 'p_CRC'

        if subtitle == 'new_test':
            df_save.to_csv(f"{output_dir}/output_data/rel_cond_mut_info_new_test.csv")
        else:
            df_save.to_csv(f"{output_dir}/output_data/rel_cond_mut_info.csv")

    # 3. PLOTTING FUNCTION
    labels = net1.get_outcome_ids("Screening") + ["Colonoscopy"]
    assign_missing_colors(labels, cfg["colors"])
    color_dict = cfg["colors"]

    def create_subplot(x_vals, mask, suffix, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        
        # Apply mask to x
        x_plot = x_vals[mask]
        
        if net2 is not None:
            # Dual net plot
            # Net 1
            y1_all, _ = dict_net[net_array[0]]
            # Net 2
            y2_all, h_y_all = dict_net[net_array[1]]
            
            # Mask data
            y1_subset = y1_all[:, mask]
            y2_subset = y2_all[:, mask]
            h_y_subset = h_y_all[mask]

            for i in range(len(labels)):
                c = color_dict.get(labels[i], '#000000')
                ax.plot(x_plot, y1_subset[i], color=c)
                ax.plot(x_plot, y2_subset[i], color=c)
                ax.fill_between(x_plot, y1_subset[i], y2_subset[i], alpha=0.1, label=labels[i], color=c)
            
            # ax.plot(x_plot, h_y_subset, label="H(CRC)", color=color_dict.get("H(CRC)", 'gray'))
            save_name = f"rel_cond_mut_info_{suffix}_bounds.png"
        else:
            # Single net plot
            y_all, h_y_all = dict_net[net_array[0]]
            y_subset = y_all[:, mask]
            h_y_subset = h_y_all[mask]
            
            for i in range(len(labels)):
                c = color_dict.get(labels[i], '#000000')
                ax.plot(x_plot, y_subset[i], label=labels[i], color=c)
            
            # ax.plot(x_plot, h_y_subset, label="H(CRC)", color=color_dict.get("H(CRC)", 'gray'))
            save_name = f"rel_cond_mut_info_{suffix}.png"

        # Extra logic for 'new_test' (only on full plot typically, or if visible)
        '''if subtitle == 'new_test' and suffix == subtitle : 
            # Check last processed net's array (matches original logic)
            arr_check = dict_net[net_array[-1]][0] 
            # Original logic: intersection of last curve and 6th curve (index 5)
            if arr_check.shape[0] > 5:
                y_last = arr_check[-1]
                y_ref = arr_check[5]
                # Manual intersection check on the plot points
                # simplified from original loop
                signs = np.sign(y_last - y_ref)
                sign_changes = ((np.roll(signs, 1) - signs) != 0) & (signs != 0)
                sign_changes[0] = False
                
                crossing_indices = np.where(sign_changes)[0]
                for idx in crossing_indices:
                     if x_values[idx] >= x_plot[0] and x_values[idx] <= x_plot[-1]:
                         ax.axvline(x=x_values[idx], ymin=0, ymax=1, color='gray', alpha=0.2, linestyle='--')
        elif suffix == subtitle and not (subtitle == 'new_test'):
             # Default vertical line
             if 0.02 >= x_plot[0] and 0.02 <= x_plot[-1]:
                 ax.axvline(x=0.02, ymin=0, ymax=1, color='gray', alpha=0.2, linestyle='--')'''

        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1), shadow=True)
        
        plt.title("Relative Reduction of Uncertainty with respect to CRC")
        
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            ax.set_xlabel("p(CRC)")
            ax.set_ylabel("RMI")

        plt.savefig(f"{output_dir}/output_images/{save_name}", bbox_inches='tight')
        plt.close(fig)

    # 4. Generate Plots
    
    # Full plot
    # Mask all
    mask_full = np.ones_like(x_values, dtype=bool)
    create_subplot(x_values, mask_full, subtitle)
    
    # Left zoom
    # Filter points somewhat within range to avoid plotting everything
    mask_left = x_values <= (zoom[0] + fine_step)
    create_subplot(x_values, mask_left, f"zoom_{subtitle}", xlim=(0, zoom[0]), ylim=(-5*step, zoom[1]))

    # Right zoom
    mask_right = x_values >= (1 - zoom[0] - fine_step)
    create_subplot(x_values, mask_right, f"zoom_2_{subtitle}", xlim=(1-zoom[0], 1), ylim=(-5*step, zoom[1]))

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



def plot_screening_counts(counts,  possible_outcomes, operational_limit = None, counts_w_lim = None, log_dir = None, lambda_list = None, label = '_', timestamp = None):    
    fig, ax = plt.subplots()
    width = 0.25
    x = np.arange(len(possible_outcomes))

    # Loop through each bar to add the text annotations and apply conditional styling
    for i, outcome in enumerate(possible_outcomes):
        count = counts.iloc[i]
        
        if counts_w_lim is not None:
            bar1 = ax.bar(x[i]-width, count, width, color='steelblue', alpha = 0.3, label = 'Recommended number of tests' if i == 0 else None) 
        else:
            
            bar1 = ax.bar(x[i], count, color='steelblue', alpha = 0.3, label = 'Recommended number of tests' if i == 0 else None)

        
        ax.text(bar1[0].get_x() + bar1[0].get_width()/2, count + 5000, str(int(count)), ha='center', color='black', fontsize=15)

        
    try: 
        for i, outcome in enumerate(possible_outcomes):
            limit = operational_limit[outcome]
            bar2 = ax.bar(x[i], limit,width, color='red', alpha = 0.3, label = 'Operational Limit' if i == 0 else None)

            try:
                ax.text(bar2[0].get_x() + bar2[0].get_width()/2, -11000, str(int(limit)), ha='center', color='red', fontsize=10)
            except:
                pass

            bar3 = ax.bar(x[i] + width, counts_w_lim.iloc[i],width, color='steelblue', label = 'Final number of tests' if i == 0 else None)
            ax.text(bar3[0].get_x() + bar3[0].get_width()/2, counts_w_lim.iloc[i] + 5000, str(int(counts_w_lim[outcome])), ha='center', color='black', fontsize=10)
    except:
        pass

    ax.legend()

    ax.set_ylim(0, 355000)
    ax.set_xticks(range(len(possible_outcomes)), possible_outcomes, rotation = 45, ha = 'right')
    ax.set_xlabel("Screening outcome")
    ax.set_ylabel("Number of tests")
    # ax.set_title("Recommended Tests vs. Operational Limit")

    if lambda_list is not None:
        ax.text(0.7, 0.8, r"$\lambda_1 =" + f"{lambda_list[0]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)
        ax.text(0.7, 0.75, r"$\lambda_2 =" + f"{lambda_list[1]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)
        ax.text(0.7, 0.7, r"$\lambda_3 =" + f"{lambda_list[2]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)
        ax.text(0.7, 0.65, r"$\lambda_4 =" + f"{lambda_list[3]:,.2f}" + "$", color='black', fontsize=9,
            ha='left', va='center', transform=ax.transAxes)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%H-%M-%S")

    plt.savefig(f"{log_dir}/screening_counts_{label}_{timestamp}.png", bbox_inches='tight')
    plt.close(fig)

    return