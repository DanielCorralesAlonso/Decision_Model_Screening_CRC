import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid = 'ignore')

from update import update_influence_diagram

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

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
parser.add_argument('--elicit', type=bool, default=cfg["elicit"], help='Elicitation method, choose between option1 or tanh')
parser.add_argument('--new_test', type=bool, default=cfg["new_test"], help='New test to be added to the model')

# Parse the arguments
args = parser.parse_args()


# Update the influence diagram
net = update_influence_diagram(
    model_type = args.model_type,
    value_function = args.value_function,
    elicit = args.elicit,
    ref_patient_chars = cfg["patient_chars"],
    new_test = args.new_test,
    logger = logger
)