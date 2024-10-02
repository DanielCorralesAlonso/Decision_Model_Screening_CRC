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
from elicitation import parameter_elicitation_utilities_linear, parameter_elicitation_utilities_tanh

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
    
    try:
        net.delete_arc("Results_of_Screening", "Colonoscopy")
        net.update_beliefs()
    except:
        pass

    net.update_beliefs()

    vars = np.array(["No scr", "Colonoscopy", "gFOBT", "Colonoscopy", "FIT", "Colonoscopy", "Blood_test", "Colonoscopy", "sDNA", "Colonoscopy", "CTC", "Colonoscopy", "CC", "Colonoscopy"])
    comf_levels = np.array([4,1,3,1,3,1,3,1,3,1,2,1,2,1])
    # util_levels = np.array(net.get_node_value("Value_of_CRC_detection_by_screening") + net.get_node_value("Value_of_CRC_detection_by_colonoscopy"))
    util_levels = np.array(net.get_node_value("INFO"))

    cost_levels = np.array(net.get_node_value("Cost_of_Screening"))
    # cost_levels = np.concatenate((cost_levels[::2], [0, cost_levels[1]]), axis = 0)

    lambda_list = []
    for i in range(1,5):
        lambda_k_list = []
        sel_vars = np.unique(vars[comf_levels == i] ) 
        
        logging.info("#################################################")
        logging.info(f"Comfort level {i}: {sel_vars}")

        # pdb.set_trace()
        if len(sel_vars) > 1 and i != 4:
            for comb in combinations(sel_vars, 2):
                

                logging.info(f"{comb[0]:<10} || Comfort: {comf_levels[vars == comb[0]].item():<4}| Value of info: {util_levels[vars == comb[0]].item():<6.3f}| Cost: {cost_levels[vars == comb[0]].item():<6.2f}| log10(Cost): {np.log10(cost_levels[vars == comb[0]].item()):<6.3f}|") 
                logging.info(f"{comb[1]:<10} || Comfort: {comf_levels[vars == comb[1]].item():<4}| Value of info: {util_levels[vars == comb[1]].item():<6.3f}| Cost: {cost_levels[vars == comb[1]].item():<6.2f}| log10(Cost): {np.log10(cost_levels[vars == comb[1]].item()):<6.3f}|")

                current_lambda = (np.log10(cost_levels[vars == comb[0]].item()) - np.log10(cost_levels[vars == comb[1]].item()))/(util_levels[vars == comb[0]].item() - util_levels[vars == comb[1]].item())

                if (cost_levels[vars == comb[0]].item() <= cost_levels[vars == comb[1]].item()) and (util_levels[vars == comb[0]].item() > util_levels[vars == comb[1]].item()):
                    logging.info(f"{comb[0]} must be preferred over {comb[1]} ! Thus new price for {comb[1]} must be cheaper than price for {comb[0]}")
                    preference = comb[0]
                    unpreferred = comb[1]

                    if current_lambda < lambda_list[-1]:
                        logging.info(f"Furthermore, for consistency, the new cost of {comb[1]} must be smaller than {np.power(10, np.log10(cost_levels[vars == comb[0]].item()) - lambda_list[-1] * (util_levels[vars == comb[0]].item() - util_levels[vars == comb[1]].item())):.2f} €")
                        logging.info(f"Keep in mind this is just for guidance and the new cost may not be that close to this value")


                elif (cost_levels[vars == comb[0]].item() >= cost_levels[vars == comb[1]].item()) and (util_levels[vars == comb[0]].item() < util_levels[vars == comb[1]].item()):
                    logging.info(f"{comb[1]} must be preferred over {comb[0]} ! Thus new price for {comb[0]} must be cheaper than price for {comb[1]}")
                    preference = comb[1]
                    unpreferred = comb[0]

                    if current_lambda < lambda_list[-1]:
                        logging.info(f"Furthermore, for consistency, the new cost of {comb[0]} must be smaller than {np.power(10, np.log10(- cost_levels[vars == comb[1]].item()) + lambda_list[-1] * (util_levels[vars == comb[0]].item() - util_levels[vars == comb[1]].item())):.2f} €")
                        logging.info(f"Keep in mind this is just for guidance and the new cost may not be that close to this value")


                else:

                    cost_opt_greater = max(cost_levels[vars == comb[0]].item(), cost_levels[vars == comb[1]].item())
                    cost_opt_lower = min(cost_levels[vars == comb[0]].item(), cost_levels[vars == comb[1]].item())

                    if cost_opt_greater == cost_levels[vars == comb[0]].item():
                        option_greater = comb[0]
                        option_lower = comb[1]
                    else:
                        option_greater = comb[1]
                        option_lower = comb[0]

                    info_opt_greater = max(util_levels[vars == comb[0]].item(), util_levels[vars == comb[1]].item())
                    info_opt_lower = min(util_levels[vars == comb[0]].item(), util_levels[vars == comb[1]].item())

                    #if we prefer option_greater, then the cost of option_lower must decrease and thus we will look for a lambda_k which is greater than the current lambda
                    lambda_max = (np.log10(cost_opt_greater))/ (info_opt_greater - info_opt_lower)
                    # if we prefer option_lower, then the cost of option_greater must decrease and thus we will look for a lambda_k which is smaller than the current lambda
                    lambda_min = lambda_list[-1]
                    logging.info(f"Lambda_k: {lambda_min} < lambda_k < {lambda_max}")
                    logging.info(f"Current lambda: {current_lambda}")
                    

                    if current_lambda < lambda_min:
                        logging.info(f"To be consistent with the previous answers the user must prefer {option_greater}")
                        preference = option_greater

                        max_new_cost = np.power(10, np.log10(cost_opt_greater) - lambda_min * (info_opt_greater - info_opt_lower))
                        logging.info(f"Furthermore, the new cost of {option_lower} must be smaller than {max_new_cost:.2f} €")

                    else: 
                        logging.info(f"No consistency issues so far")
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
            logging.info(f"{var:<5} || Comfort: {np.unique(comf_levels[vars == var]).item():<4}| Value of info: {util_levels[vars == var][0]:<6.3f}| Cost: {cost_levels[vars == var][0]:<6.2f}| log10(Cost): {np.log10(cost_levels[vars == var][0]):<6.3f}|")

            synthetic_info = 0.4
            logging.info(f"We need you to give us the cost for a synthetic option with info {synthetic_info}, for which you would be indifferent between colonoscopy and the synthetic option")
            logging.info(f"User info for synthetic option: {synthetic_info}")
            synthetic_indiff_cost = float(input("---> Cost? Please insert a number: "))
            logging.info(f"User cost for synthetic option: {synthetic_indiff_cost}")

            lambda_k = (np.log10(cost_levels[vars == sel_vars[0]].item()) - np.log10(synthetic_indiff_cost))/(util_levels[vars == sel_vars[0]].item() - synthetic_info)
            logging.info(f"Lambda {i}: {lambda_k}")
            lambda_k_list.append(lambda_k)

        elif i == 4:
            logging.info(f"Setting lambda for the last comfort level to {np.ceil(lambda_list[-1])}")
            lambda_k_list = np.ceil(lambda_list[-1])   # Take smallest integer greater than lambda_approx
        


        lambda_approx = np.median(lambda_k_list)
        logging.info(f"Average lambda for comfort level {i}: {np.mean(lambda_k_list)}")
        logging.info(f"Median lambda for comfort level {i}: {np.median(lambda_k_list)}")

        lambda_list.append(lambda_approx)


    logging.info(f"Elicitation is done! The elicited lambda values are: {lambda_list}")

    if sorted(lambda_list) != lambda_list:
        logging.warning("Lambda values are not in increasing order! Model assumptions for comfort levels are not met!")
        exit()


    lambdas = np.array([lambda_list[3], lambda_list[0], lambda_list[2], lambda_list[0], 
                        lambda_list[2], lambda_list[0], lambda_list[2], lambda_list[0], 
                        lambda_list[2], lambda_list[0], lambda_list[1], lambda_list[0], lambda_list[1], lambda_list[0],])


    try:
        net.add_arc("Results_of_Screening", "Colonoscopy")
        net.update_beliefs()
    except:
        pass

    return lambdas