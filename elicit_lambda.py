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

value_function = "rel_pcmi"

# ----------------------------------------------------------------------
# Read model type from input in the command line
import sys
if len(sys.argv) > 1:
    model_type = sys.argv[1]
else:
    model_type = "linear" # "tanh" or "linear"
# ----------------------------------------------------------------------

print("Model type: ", model_type)

# Read the network -----------------------------------------------------
print("Reading network...")
net = pysmile.Network()
net.read_file(f"decision_models/DM_screening_rel_pcmi_{model_type}.xdsl")
# ----------------------------------------------------------------------


net.clear_all_evidence()
net.set_evidence("Age", "age_5_old_adult")
net.set_evidence("Sex", "M")
# net.set_evidence("SES", "ses_0")  # not mandatory
net.set_evidence("SD", "SD_1_short")
net.set_evidence("PA", "PA_1")
net.set_evidence("Smoking", "sm_3_ex_smoker")
#net.set_evidence("Depression", False)  # not mandatory
#net.set_evidence("Anxiety", False)  # not mandatory
net.set_evidence("BMI", "bmi_3_overweight")
net.set_evidence("Alcohol", "low")
#net.set_evidence("Hypertension", False)  # not mandatory
#net.set_evidence("Diabetes", False)  # not mandatory
#net.set_evidence("Hyperchol_", False)  # not mandatory

net.update_beliefs()




# Get all combinations of screening methods with the same level of comfort
# Use itertools


vars = np.array(["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC", "No colonoscopy", "Colonoscopy"])
comf_levels = np.array([4,3,3,3,3,2,2,4,1])
util_levels = np.array(net.get_node_value("Value_of_CRC_detection_by_screening") + net.get_node_value("Value_of_CRC_detection_by_colonoscopy"))

cost_levels = np.array(net.get_node_value("Cost_of_Screening"))
cost_levels = np.concatenate((cost_levels[::2], [0, cost_levels[-1]]), axis = 0)

comfort_level_dict = dict(zip(vars, comf_levels))
utility_level_dict = dict(zip(vars, util_levels))

lambda_list = []
for i in range(1,5):
    lambda_k_list = []
    sel_vars = vars[comf_levels == i]  
    
    print(f"Comfort level {i}: {sel_vars}")
    if len(sel_vars) > 1:
        for comb in combinations(sel_vars, 2):

            print(comb)
            print(comb[0], "Comfort:", comf_levels[vars == comb[0]].item(), "; Value of info:", util_levels[vars == comb[0]].item(), "; Cost:", cost_levels[vars == comb[0]].item()) 
            print(comb[1], "Comfort:", comf_levels[vars == comb[1]].item() , "; Value of info:", util_levels[vars == comb[1]].item(), "; Cost:", cost_levels[vars == comb[1]].item())


            if (cost_levels[vars == comb[0]].item() <= cost_levels[vars == comb[1]].item()) and (util_levels[vars == comb[0]].item() > util_levels[vars == comb[1]].item()):
                print(f"{comb[0]} is preferred over {comb[1]}")
                preference = comb[0]
            elif (cost_levels[vars == comb[0]].item() >= cost_levels[vars == comb[1]].item()) and (util_levels[vars == comb[0]].item() < util_levels[vars == comb[1]].item()):
                print(f"{comb[1]} is preferred over {comb[0]}")
                preference = comb[1]
            else:
                preference = input("Which one do you prefer?")
            
            new_cost = float(input("What would the cost of the non-preferred option need to be for you to be indifferent between the two options?"))


            if preference == comb[0]:
                lambda_k = (np.log10(cost_levels[vars == comb[0]].item()) - np.log10(new_cost))/(util_levels[vars == comb[0]].item() - util_levels[vars == comb[1]].item())
                print(f"Lambda {i}:", lambda_k)
            elif preference == comb[1]:
                lambda_k = (np.log10(cost_levels[vars == comb[1]].item()) - np.log10(new_cost))/(util_levels[vars == comb[1]].item() - util_levels[vars == comb[0]].item())
                print(f"Lambda {i}:", lambda_k)

            lambda_k_list.append(lambda_k)

    elif i == 1:
        synthetic_info = 0.5
        synthetic_indiff_cost = 600

        lambda_k = (np.log10(cost_levels[vars == sel_vars[0]].item()) - np.log10(synthetic_indiff_cost))/(util_levels[vars == sel_vars[0]].item() - synthetic_info)
        print(f"Lambda {i}:", lambda_k)
        lambda_k_list.append(lambda_k)
    
    else:
        print(sel_vars[0], "Comfort:", comf_levels[vars == sel_vars[0]].item(), "; Value of info:", util_levels[vars == sel_vars[0]].item())


    lambda_approx = np.mean(lambda_k_list)
    print(f"Average lambda for comfort level {i}:", lambda_approx)

    lambda_list.append(lambda_approx)


print("Elicitation is done! The elicited lambda values are:", lambda_list)


arr_comft = np.array([lambda_list[3], lambda_list[0], lambda_list[2], lambda_list[0], 
                      lambda_list[2], lambda_list[0], lambda_list[2], lambda_list[0], 
                      lambda_list[2], lambda_list[0], lambda_list[1], lambda_list[0], lambda_list[1], lambda_list[0],])


net.set_node_definition("Value_of_comfort", arr_comft)
net.set_mau_expressions(node_id = "V", expressions = [f"Value_of_comfort*VALUE - Log10(COST+1)"])

print("Saving network...")
net.write_file(f"decision_models/DM_screening_{value_function}_{model_type}.xdsl")