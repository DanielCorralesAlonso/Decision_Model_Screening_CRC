import pysmile
import pysmile_license
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from preprocessing import preprocessing

from network_functions import calculate_network_utilities, new_screening_strategy, old_screening_strategy, create_folders_logger
from elicitation import parameter_elicitation_utilities_linear
from network_functions import create_folders_logger
from use_case_new_strategy import use_case_new_strategy

import pdb

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)


def full_example(only_counts = False):
    if cfg["new_test"] == True:
        file_location = "decision_models/DM_screening_rel_point_cond_mut_info_linear_new_test.xdsl"
    else:
        file_location = "decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl"

    df_test = pd.read_csv("private/df_2016.csv")
    df_test = preprocessing(df_test)
    df_test = df_test.rename(columns = {"Hyperchol.": "Hyperchol_"})


    net = pysmile.Network()
    net.read_file(file_location)

    net.update_beliefs()
    rho_comfort = net.get_node_value("Value_of_comfort")[2]

    PE_info_array = np.array( cfg["full_example"]["PE_info_array"] )
    PE_cost_array = np.array( cfg["full_example"]["PE_cost_array"] )


    single_run = cfg['single_run']
    best_f1_score = {}

    logger, log_dir = create_folders_logger(single_run = single_run, label="use_case_sens_analysis_")
    logger.info("Starting full sensitivity analysis with classification...")

    fig, axes = plt.subplots(len(PE_info_array), len(PE_cost_array), figsize=(16, 16))

    for i, param1 in enumerate(PE_info_array):
        for j, param2 in enumerate(PE_cost_array):
            logger.info(f"PE_info: {param1}, PE_cost: {param2}, STARTING...")
            # Call the custom function with the current combination of parameters
            net.clear_all_evidence()
            params = parameter_elicitation_utilities_linear(net,PE = 0.7, PE_info = param1, PE_cost = param2, rho_comfort = rho_comfort, value_function = "rel_point_cond_mut_info", logging = None)
            logger.info(f"Risk aversion params: {params}")
            net.set_mau_expressions(node_id = "U", expressions = [f"{params[0]} - {params[1]}*Exp( - {params[2]} * V)"])
            net.update_beliefs()

            _ , counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = True)

            #code from plots.py function plot_screening_counts()
            bars1 = axes[i,j].bar(possible_outcomes, counts, color = 'steelblue', alpha= 0.3, label = 'Number of tests')
            for bar in bars1:
                axes[i,j].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000, str(bar.get_height()), ha='center', color='black', fontsize=15)

            axes[i,j].legend()

            axes[i,j].set_ylim(0, 350000)
            axes[i,j].set_xticks(range(len(possible_outcomes)), possible_outcomes, rotation = 45)
            axes[i,j].set_xlabel("Screening outcome")
            axes[i,j].set_ylabel("Number of tests")
            axes[i,j].set_title("PE_info = " + str(param1) + " and PE_cost = " + str(param2))



            logger.info(f"PE_info: {param1}, PE_cost: {param2}, DONE!")

            plt.tight_layout()  
            if cfg["new_test"] == True:
                plt.savefig(f"{log_dir}/sens_analysis_screening_counts_new_test.png")
            else:
                plt.savefig(f"{log_dir}/sens_analysis_screening_counts.png")

            #create folder run_label
            import os
            run_label = f"PE_info_{i}_PE_cost_{j}"
            log_dir_ext = f"{log_dir}/{run_label}"
            os.makedirs(log_dir_ext, exist_ok=True)
            net.write_file(f"{log_dir_ext}/DM_screening.xdsl")

            # create a df with possible_outcomes and counts
            df_counts = pd.DataFrame({"possible_outcomes": possible_outcomes, "counts": counts})
            df_counts.to_csv(f"{log_dir_ext}/df_counts.csv")

            if not only_counts:
                run_label = f"PE_info_{i}_PE_cost_{j}"
                

                f1_score = use_case_new_strategy(
                    net = net,
                    file_location = file_location,
                    single_run = single_run,
                    use_case_new_test= cfg["new_test"],
                    all_variables= True,
                    logger = logger,
                    log_dir = log_dir,
                    run_label = run_label,
                    best_f1_score= best_f1_score
                    )
                logger.info(f"F1 score for PE_info = {param1} and PE_cost = {param2}: {best_f1_score}")
                best_f1_score.update(f1_score)

    

    best_f1_score = pd.DataFrame(best_f1_score)
    best_f1_score.to_csv(f"{log_dir}/best_f1_score.csv")
    best_params = best_f1_score.idxmax(axis=1)
    logger.info(f"Best model: {best_params}")




    plt.close(fig)



if __name__ == "__main__":
    full_example(only_counts=True)





