# elicit_lambda.py
import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools
from itertools import combinations
from plots import plot_cond_mut_info, plot_relative_cond_mut_info
from save_info_values import save_info_values
np.seterr(divide='ignore', invalid = 'ignore')

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures
from functions import system_of_eq, tanh_fun
from elicitation import parameter_elicitation_utilities_option1, parameter_elicitation_utilities_tanh

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
import pdb



# Get all combinations of screening methods with the same level of comfort
# Use itertools

def elicit_lambda(patient_chars, net, logging):
    net.clear_all_evidence()

    for key, value in patient_chars.items():
        net.set_evidence(key, value)
    
    net.update_beliefs()

    vars = np.array(["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC", "No colonoscopy", "Colonoscopy"])
    comf_levels = np.array([4,3,3,3,3,2,2,4,1])
    util_levels = np.array(net.get_node_value("Value_of_CRC_detection_by_screening") + net.get_node_value("Value_of_CRC_detection_by_colonoscopy"))

    cost_levels = np.array(net.get_node_value("Cost_of_Screening"))
    cost_levels = np.concatenate((cost_levels[::2], [0, cost_levels[1]]), axis = 0)

    lambda_list = []
    for i in range(1,5):
        lambda_k_list = []
        sel_vars = vars[comf_levels == i]  
        
        logging.info("#################################################")
        logging.info(f"Comfort level {i}: {sel_vars}")
        if len(sel_vars) > 1:
            for comb in combinations(sel_vars, 2):

                logging.info(f"{comb[0]:<10} || Comfort: {comf_levels[vars == comb[0]].item():<4}| Value of info: {util_levels[vars == comb[0]].item():<6.3f}| Cost: {cost_levels[vars == comb[0]].item():<6.2f}| log10(Cost): {np.log10(cost_levels[vars == comb[0]].item()):<6.3f}|") 
                logging.info(f"{comb[1]:<10} || Comfort: {comf_levels[vars == comb[1]].item():<4}| Value of info: {util_levels[vars == comb[1]].item():<6.3f}| Cost: {cost_levels[vars == comb[1]].item():<6.2f}| log10(Cost): {np.log10(cost_levels[vars == comb[1]].item()):<6.3f}|")


                if (cost_levels[vars == comb[0]].item() <= cost_levels[vars == comb[1]].item()) and (util_levels[vars == comb[0]].item() > util_levels[vars == comb[1]].item()):
                    logging.info(f"{comb[0]} must be preferred over {comb[1]} ! Thus new price for {comb[1]} must be cheaper than price for {comb[0]}")
                    preference = comb[0]
                    unpreferred = comb[1]
                elif (cost_levels[vars == comb[0]].item() >= cost_levels[vars == comb[1]].item()) and (util_levels[vars == comb[0]].item() < util_levels[vars == comb[1]].item()):
                    logging.info(f"{comb[1]} must be preferred over {comb[0]} ! Thus new price for {comb[0]} must be cheaper than price for {comb[1]}")
                    preference = comb[1]
                    unpreferred = comb[0]
                else:
                    preference = input(f"---> Which one do you prefer? Choose between {comb[0]} and {comb[1]}: ")
                    logging.info(f"Preference of the user is: {preference}")
                    unpreferred = comb[0] if preference == comb[1] else comb[1]
                
                new_cost = float(input(f"---> What would the cost of the non-preferred option need to be for you to be indifferent between the two options? Please insert a number: "))
                logging.info(f"New cost for {unpreferred}: {new_cost}")

                if preference == comb[0]:
                    lambda_k = (np.log10(cost_levels[vars == comb[0]].item()) - np.log10(new_cost))/(util_levels[vars == comb[0]].item() - util_levels[vars == comb[1]].item())
                    logging.info(f"Lambda {i}: {lambda_k}")
                elif preference == comb[1]:
                    lambda_k = (np.log10(cost_levels[vars == comb[1]].item()) - np.log10(new_cost))/(util_levels[vars == comb[1]].item() - util_levels[vars == comb[0]].item())
                    logging.info(f"Lambda {i}: {lambda_k}")

                lambda_k_list.append(lambda_k)
                logging.info("-------------------------------------------------")

        if i == 1:
            var = "Colonoscopy"
            logging.info(f"{var:<5} || Comfort: {comf_levels[vars == var].item():<4}| Value of info: {util_levels[vars == var].item():<6.3f}| Cost: {cost_levels[vars == var].item():<6.2f}| log10(Cost): {np.log10(cost_levels[vars == var].item()):<6.3f}|")

            logging.info("We need you to give us a level of info and cost for a synthetic option that you would be indifferent between colonoscopy and the synthetic option.")
            synthetic_info = float(input("---> Info? Please insert a number: "))
            logging.info(f"User info for synthetic option: {synthetic_info}")
            synthetic_indiff_cost = float(input("---> Cost? Please insert a number: "))
            logging.info(f"User cost for synthetic option: {synthetic_indiff_cost}")

            lambda_k = (np.log10(cost_levels[vars == sel_vars[0]].item()) - np.log10(synthetic_indiff_cost))/(util_levels[vars == sel_vars[0]].item() - synthetic_info)
            logging.info(f"Lambda {i}: {lambda_k}")
            lambda_k_list.append(lambda_k)

        elif i == 4:
            # lambda_k_list = 10
            lambda_k_list = np.ceil(lambda_list[-1])   # Take smallest integer greater than lambda_approx
        


        lambda_approx = np.mean(lambda_k_list)
        logging.info(f"Average lambda for comfort level {i}: {lambda_approx}")
        logging.info(f"Median lambda for comfort level {i}: {np.median(lambda_k_list)}")

        lambda_list.append(lambda_approx)


    logging.info(f"Elicitation is done! The elicited lambda values are: {lambda_list}")

    if sorted(lambda_list) != lambda_list:
        logging.warning("Lambda values are not in increasing order! Model assumptions for comfort levels are not met!")
        exit()


    lambdas = np.array([lambda_list[3], lambda_list[0], lambda_list[2], lambda_list[0], 
                        lambda_list[2], lambda_list[0], lambda_list[2], lambda_list[0], 
                        lambda_list[2], lambda_list[0], lambda_list[1], lambda_list[0], lambda_list[1], lambda_list[0],])


    return lambdas