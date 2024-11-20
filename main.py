import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid = 'ignore')

from update import update_influence_diagram
from plots import plot_relative_cond_mut_info

import logging
import datetime 
import os

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
import pdb

import argparse

current_dir = os.getcwd()
log_dir = os.path.join(current_dir, 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

date_str = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join(current_dir, 'logs', date_str)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.datetime.now().strftime("%H-%M-%S")
log_filename = os.path.join(log_dir, f"decision_model_{timestamp}.log")

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


parser = argparse.ArgumentParser(description="Update the influence diagram")
    
# Introduction of arguments
parser.add_argument('--model_type', type=str, default=cfg["model_type"], help='Model type, choose between linear or tanh')
parser.add_argument('--value_function', type=str, default= cfg["value_function"], help='Value function, choose between rel_pcmi or pcmi')
parser.add_argument('--elicit', type=bool, default=cfg["elicit"], help='Elicitation method, choose between linear or tanh')
parser.add_argument('--noise', type =bool, default=cfg["noise"], help='Add noise to the current lambdas for comfort')
parser.add_argument('--calculate_info_values', type=bool, default=cfg["calculate_info_values"], help='Calculate the information values')
parser.add_argument('--new_test', type=bool, default=cfg["new_test"], help='New test to be added to the model')
parser.add_argument('--sens_analysis_metrics', type=bool, default=False, help='Perform sensitivity analysis with respect to the performance metrics of the screening methods')

# Parse the arguments
args = parser.parse_args()

def main(args):

    # Normal update.
    if not args.sens_analysis_metrics:

        logger.info("----- Normal update of the influence diagram -----")
        output_dir = f"outputs/{args.model_type}_{args.value_function}_elicit{args.elicit}_newtest{args.new_test}"

        net = update_influence_diagram(
            model_type = args.model_type,
            value_function = args.value_function,
            elicit = args.elicit,
            noise = args.noise,
            calculate_info_values=args.calculate_info_values,
            ref_patient_chars = cfg["patient_chars"],
            new_test = args.new_test,
            logger = logger,
            output_dir = output_dir
        )

        # ----------------------------------------------------------------------
        logger.info("Done!")
        # ----------------------------------------------------------------------
        

    # Sensitivity analysis with respect to the performance metrics of the screening methods.
    else:

        logger.info("----- Update the influence diagram with lower and upper bounds on the performance metrics -----")
        output_dir = f"outputs/{args.model_type}_{args.value_function}_elicit{args.elicit}_newtest{args.new_test}_sens"

        net_lower = update_influence_diagram(
            model_type = args.model_type,
            value_function = args.value_function,
            elicit = args.elicit,
            noise = args.noise,
            ref_patient_chars = cfg["patient_chars"],
            new_test = args.new_test,
            sens_analysis_metrics = "lower", # Lower bound
            logger = logger,
            output_dir = output_dir
        )

        net_upper = update_influence_diagram(
            model_type = args.model_type,
            value_function = args.value_function,
            elicit = args.elicit,
            noise = args.noise,
            ref_patient_chars = cfg["patient_chars"],
            new_test = args.new_test,
            sens_analysis_metrics = "upper", # Upper bound
            logger = logger,
            output_dir = output_dir
        )

        df_U_lower = pd.read_csv(f"{output_dir}/U_values_sens_analysis_lower.csv", index_col=0)
        df_U_upper = pd.read_csv(f"{output_dir}/U_values_sens_analysis_upper.csv", index_col=0)

        logger.info("Plotting the relative conditional mutual information bounds")
        plot_relative_cond_mut_info(net_lower, net_upper)

        df_U_lower.loc["Upper bound"] = df_U_upper.loc["U"]
        df_U_lower.rename(index={"U": "Lower bound"}, inplace=True)

        df_U_lower.to_csv(f"{output_dir}/U_values_sens_analysis.csv")

        # ----------------------------------------------------------------------
        logger.info("Done!")
        # ----------------------------------------------------------------------



if __name__ == "__main__":
    main()