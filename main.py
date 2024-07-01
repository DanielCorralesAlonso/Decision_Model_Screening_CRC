import pysmile
import pysmile_license
import numpy as np
import pandas as pd
import itertools
from plots import plot_cond_mut_info
from save_info_values import save_info_values
np.seterr(divide='ignore', invalid = 'ignore')

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures
from functions import system_of_eq, tanh_fun
from elicitation import parameter_elicitation_utilities

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
import pdb

# Read the network -----------------------------------------------------
print("Reading network...")
net = pysmile.Network()
net.read_file("genie_models/decision_models/DM_screening_submodel_tanh_rel_pcmi.xdsl")
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
print("Calculating relative pointwise conditional mutual information values...")

try:
    net.delete_arc("Results_of_Screening", "Colonoscopy")
except:
    print("No arc to delete")

value_function = "rel_pcmi"
df_value_scr, df_value_col = save_info_values(net, value_function = value_function, weighted=False)
net2 = info_value_to_net(df_value_scr, df_value_col, net)

# net2.add_arc("Results_of_Screening", "Colonoscopy")

print("Saving network...")
net2.write_file(f"genie_models/decision_models/DM_screening_submodel_tanh_{value_function}.xdsl")
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
print("Plotting info functions...")
plot_cond_mut_info(net2)
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
print("Calculating final utilities...")
params = parameter_elicitation_utilities(PE_info = cfg["PE_info"], PE_cost = cfg["PE_cost"], rho_comfort = cfg["rho_comfort"])

if params is None:
    print("Please try another initial value for the system of equations...")
    exit()

else:
    print("Parameters found: ", params)
    net2.set_mau_expressions(node_id = "U", expressions = [f"({params[0]} - {params[1]}*Exp( - {params[2]} * V)) / {params[0]}"])
    net2.write_file(f"genie_models/decision_models/DM_screening_submodel_tanh_{value_function}.xdsl")
    print("Done!")
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
print("Calculating utilities for patient X...")

# Set evidence for the patient
net2.set_evidence("Age", "age_2_young")
net2.set_evidence("Sex", "M")
net2.set_evidence("SES", "ses_0")  # not mandatory
net2.set_evidence("SD", "SD_1_short")
net2.set_evidence("PA", "PA_1")
net2.set_evidence("Smoking", "sm_1_not_smoker")
net2.set_evidence("Depression", False)  # not mandatory
net2.set_evidence("Anxiety", False)  # not mandatory
net2.set_evidence("BMI", "bmi_1_underweight")
net2.set_evidence("Alcohol", "high")
net2.set_evidence("Hypertension", False)  # not mandatory
net2.set_evidence("Diabetes", False)  # not mandatory
net2.set_evidence("Hyperchol_", False)  # not mandatory

net2.update_beliefs()

# pdb.set_trace()
if len(net2.get_node_value("U")) == 14:
    vars1 = ["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC"]
    vars2 = ["No colonoscopy", "Colonoscopy"]

    comb = list(itertools.product(vars1, vars2))

    index = pd.MultiIndex.from_tuples(comb)
    arr = np.array(net2.get_node_value("U"))

    df_U = pd.DataFrame(arr.reshape(1,-1), index=["U"], columns=index)

print(df_U)
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
print(df_U_ext)
df_U_ext.to_csv("U_values.csv")
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
print("Done!")
# ----------------------------------------------------------------------