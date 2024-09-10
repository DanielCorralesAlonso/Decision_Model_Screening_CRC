import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools
from plots import plot_cond_mut_info, plot_relative_cond_mut_info
from save_info_values import save_info_values
np.seterr(divide='ignore', invalid = 'ignore')

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures
from functions import system_of_eq, tanh_fun
from elicitation import parameter_elicitation_utilities_option1, parameter_elicitation_utilities_tanh
from elicit_lambda import elicit_lambda

import logging
import datetime 
import os

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
import pdb

# Get the current working directory
current_dir = os.getcwd()

# Define the path for the logs directory within the current directory
log_dir = os.path.join(current_dir, 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the log file path with the timestamp
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

# ----------------------------------------------------------------------
# Read model type from input in the command line
import sys
if len(sys.argv) > 2:
    model_type = sys.argv[1]
    elicit = sys.argv[2]
elif len(sys.argv) > 1 and len(sys.argv) <= 2:
    model_type = sys.argv[1]
    elicit = False  
else:
    model_type = "tanh" # "tanh" or "linear"
    elicit = False
# ----------------------------------------------------------------------

logger.info(f"Model type: {model_type}")

# Read the network -----------------------------------------------------
logger.info("Reading network...")
net = pysmile.Network()
net.read_file(f"decision_models/DM_screening_rel_pcmi_{model_type}.xdsl")
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
logger.info("Calculating relative pointwise conditional mutual information values...")

try:
    net.delete_arc("Results_of_Screening", "Colonoscopy")
except:
    logger.info("No arc to delete")

value_function = "rel_pcmi"
df_value_scr, df_value_col = save_info_values(net, value_function = value_function, weighted=False)
net2 = info_value_to_net(df_value_scr, df_value_col, net)


# ----------------------------------------------------------------------
# Define referent patient characteristics
patient_chars = {"Age": "age_3_young_adult", 
                         "Sex": "M",
                         "SD": "SD_2_normal",
                         "PA": "PA_2",
                         "Smoking": "sm_1_not_smoker",
                         "BMI": "bmi_2_normal",
                         "Alcohol": "low",
                         #"Diabetes": True,
                         #"Hypertension": True,
                         }

# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
if model_type == "tanh":
    logging.info("Defining value of comfort...")
    rho_4 = 0.6
    rho_3 = 0.55
    rho_2 = 0.50
    rho_1 = 0.40
    arr_comft = np.array([rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4, rho_4,
                        rho_4, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_2, rho_1, rho_2, rho_1, # Added discomfort when colonoscopy is not mandatory
                        rho_4, rho_4, rho_3, rho_3, rho_3, rho_3, rho_3, rho_3, rho_3, rho_3, rho_2, rho_2, rho_2, rho_2,]) #No added discomfort when colonoscopy is mandatory
    net2.set_node_definition("Value_of_comfort", arr_comft)

    net2.set_mau_expressions(node_id = "V", expressions = [f"((8131.71-COST)/8131.71)*Tanh(VALUE*Value_of_comfort)"])

elif model_type == "linear":
    logger.info("Eliciting value of comfort...")

    if elicit == "elicit":

        lambdas = elicit_lambda(patient_chars = patient_chars,
                                net = net2, logging = logger)
        
        net.set_node_definition("Value_of_comfort", lambdas)
        net.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*VALUE - Log10(COST+1)"])

    else:
        rho_4 = 8
        rho_3 = 6.5
        rho_2 = 6.25
        rho_1 = 6 # Not a bad choice of values. Try to justify them correctly.
        arr_comft = np.array([rho_4, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_3, rho_1, rho_2, rho_1, rho_2, rho_1,])
        net2.set_node_definition("Value_of_comfort", arr_comft)

        net2.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*VALUE - Log10(COST+1)"])
        

        lambdas = net2.get_node_value("Value_of_comfort")

        logger.info("No elicitation of lambda values, taking default values...")




# ----------------------------------------------------------------------
logger.info("Saving network...")
net2.write_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
logger.info("Plotting info functions...")
plot_cond_mut_info(net2)
plot_relative_cond_mut_info(net2, subtitle = '', zoom = (0.001, 0.1))
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
logger.info("Calculating final utilities...")

if model_type == "tanh":
    params = parameter_elicitation_utilities_tanh(PE_info = cfg["PE_info"], PE_cost = cfg["PE_cost"], rho_comfort = lambdas[2])
elif model_type == "linear":
    params = parameter_elicitation_utilities_option1(PE_info = cfg["PE_info"], PE_cost = cfg["PE_cost"], rho_comfort = lambdas[2], logging = logger)

if params is None:
    logger.warning("Please try another initial value for the system of equations...")
    exit()

else:
    logger.info("Parameters found: {params}")
    net2.set_mau_expressions(node_id = "U", expressions = [f"({params[0]} - {params[1]}*Exp( - {params[2]} * V))"])
    net2.write_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")
    logger.info("Done!")
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
logger.info("Calculating utilities for patient X...")

net.clear_all_evidence()

for key, value in patient_chars.items():
    net.set_evidence(key, value)

net.update_beliefs()

# pdb.set_trace()
if len(net2.get_node_value("U")) == 14:
    vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC"]
    vars2 = ["No colonoscopy", "Colonoscopy"]

    comb = list(itertools.product(vars1, vars2))

    index = pd.MultiIndex.from_tuples(comb)
    arr = np.array(net2.get_node_value("U"))

    df_U = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)

logger.info(df_U)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
net2.add_arc("Results_of_Screening", "Colonoscopy")
net2.update_beliefs()

vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC"]
vars2 = ["No pred", "Pred False", "Pred True"]
vars3 = ["No colonoscopy", "Colonoscopy"]

comb = list(itertools.product(vars1, vars2, vars3))

index = pd.MultiIndex.from_tuples(comb)
arr = np.array(net2.get_node_value("U"))

df_U_ext = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)
logger.info(df_U_ext)
df_U_ext.to_csv("U_values.csv")
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
logger.info("Done!")
# ----------------------------------------------------------------------