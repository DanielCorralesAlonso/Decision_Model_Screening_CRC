from elicitation import parameter_elicitation_utilities_linear
import numpy as np
import matplotlib.pyplot as plt

import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures, cond_kl_divergence
from save_info_values import save_info_values
from plots import plot_cond_mut_info, plot_relative_cond_mut_info

from preprocessing import preprocessing

from network_functions import calculate_network_utilities, new_screening_strategy, old_screening_strategy, compare_strategies
from simulations import plot_classification_results
from plots import plot_estimations_w_error_bars, plot_screening_counts

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

import logging
import datetime 
import os

np.seterr(divide='ignore', invalid = 'ignore')


current_dir = os.getcwd()
log_dir = os.path.join(current_dir, 'logs')
# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

date_str = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join(log_dir, date_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

time_str = datetime.datetime.now().strftime("%H-%M-%S")
log_dir = os.path.join(log_dir, "sens_analysis_" + time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

new_test = cfg["use_case_new_test"]
timestamp = datetime.datetime.now().strftime("%H-%M-%S")
log_filename = os.path.join(log_dir, f"sens_analysis_{timestamp}_new_test_{new_test}.log")

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the minimum logging level

# Create handlers for file and console
file_handler = logging.FileHandler(log_filename)  # Logs to file
console_handler = logging.StreamHandler()  # Logs to console

# Set the logging level for both handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Define the formatter for logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add the formatter to the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)




if cfg["use_case_new_test"] == True:
    file_location = "outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestTrue/decision_models/DM_screening_rel_point_cond_mut_info_linear_new_test.xdsl"
else:
    file_location = "outputs/linear_rel_point_cond_mut_info_elicitFalse_newtestFalse/decision_models/DM_screening_rel_point_cond_mut_info_linear.xdsl"

net = pysmile.Network()
net.read_file(file_location)


PE_info_array = np.array([4, 4.2 , 4.4, 4.5,])
PE_cost_array = np.array([ 5, 10, 50, 100, 500])

logger.info(f"PE_info_array: {PE_info_array}")
logger.info(f"PE_cost_array: {PE_cost_array}")

net.update_beliefs()
rho_comfort = net.get_node_value("Value_of_comfort")[2]

df_test = pd.read_csv("private/df_2016.csv")
df_test = preprocessing(df_test)
df_test = df_test.rename(columns = {"Hyperchol.": "Hyperchol_"})


def sens_analysis_PE_method(net, df_test, PE_info_array, PE_cost_array):

    # This function performs a sensitivity analysis on the PE method by varying its parameters.
    # The motivation is the depending on the relationship between PE cost and PE info, the
    # optimal screening strategy may change. We will explore how the distribution of recommended
    # screening strategies changes as we vary the parameters of the PE method.


    fig, axes = plt.subplots(len(PE_info_array), len(PE_cost_array), figsize=(16, 16))

    for i, param1 in enumerate(PE_info_array):
            for j, param2 in enumerate(PE_cost_array):
                # Call the custom function with the current combination of parameters
                params = parameter_elicitation_utilities_linear(net,PE = 0.7, PE_info = param1, PE_cost = param2, rho_comfort = rho_comfort, value_function = "rel_point_cond_mut_info", logging = None)
                logger.info(f"Params: {params}")
            
                net.set_mau_expressions(node_id = "U", expressions = [f"Max(0, Min({params[0]} - {params[1]}*Exp( - {params[2]} * V), 1))"])
                net.update_beliefs()

                _ , counts, possible_outcomes = calculate_network_utilities(net, df_test, full_calculation = True)

                #code from plots.py function plot_screening_counts()
                bars1 = axes[i,j].bar(possible_outcomes, counts, color = 'blue', label = 'Number of tests')
                for bar in bars1:
                    axes[i,j].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000, str(bar.get_height()), ha='center', color='black', fontsize=10)

                axes[i,j].legend()

                axes[i,j].set_ylim(0, 350000)
                axes[i,j].set_xticks(range(len(possible_outcomes)), possible_outcomes, rotation = 45)
                axes[i,j].set_xlabel("Screening outcome")
                axes[i,j].set_ylabel("Number of tests")
                axes[i,j].set_title("PE_info = " + str(param1) + " and PE_cost = " + str(param2))



                logger.info(f"PE_info: {param1}, PE_cost: {param2}, DONE!")

                plt.tight_layout()  
                if cfg["use_case_new_test"] == True:
                    plt.savefig(f"{log_dir}/sens_analysis_screening_counts_new_test.png")
                else:
                    plt.savefig(f"{log_dir}/sens_analysis_screening_counts.png")


    
    plt.close(fig)


def main():
    sens_analysis_PE_method(net,df_test, PE_info_array, PE_cost_array)

if __name__ == "__main__":
    main()
# Compare this snippet from sens_analysis_PE_method.py: 


