
import pysmile
import pysmile_license
import numpy as np
import pandas as pd

from preprocessing import preprocessing


# ----------------------------------------------------------------------
# Read model type from input in the command line
import sys
if len(sys.argv) > 1:
    calculate = sys.argv[1]
else:
    calculate = False
# ----------------------------------------------------------------------

model_type = "linear"

net = pysmile.Network()
net.read_file(f"decision_models/DM_screening_rel_pcmi_{model_type}.xdsl")

df_test = pd.read_csv("private/df_2016.csv")
df_test = preprocessing(df_test)

df_test = df_test.rename(columns = {"Hyperchol.": "Hyperchol_"})

y = np.array(list(df_test["CRC"]*1))

vars1 = np.array(["No scr", "gFOBT", "FIT", "Blood_test", "sDNA", "CTC", "CC", "Colonoscopy"])

if calculate == "calculate":

    df_utilities_2016 = pd.DataFrame([], columns = vars1)

    for i in range(df_test.shape[0]):
        sample = df_test.iloc[i].drop(labels = ["CRC"])
        sample_dict = sample.to_dict() 

        net.clear_all_evidence()

        for key, value in sample.items():
            if type(value) == np.bool_:
                net.set_evidence(key, bool(value))
            else:
                net.set_evidence(key, value)

        net.update_beliefs()

        arr = np.array(net.get_node_value("U"))

        U = np.concatenate((arr[::2], [arr[1]]), axis = 0)

        row = pd.DataFrame(U.reshape(1,8), columns=vars1)
        df_utilities_2016 = pd.concat([df_utilities_2016, row], ignore_index = True)

    df_utilities_2016 = pd.concat( [df_utilities_2016, df_test["CRC"] ] , axis = 1)
    df_utilities_2016.to_csv("utilities_2016.csv", index = False)

else: 
    df_utilities_2016 = pd.read_csv("utilities_2016.csv")



df_selected_current_strategy = df_test[df_test["Age"] == "age_5_old_adult"]
tot_num_patients = df_selected_current_strategy.shape[0]   
print("------------------------")
print("Number of selected patients by current strategy is: ", tot_num_patients)
print("Number of patients with CRC by current strategy is: ", df_selected_current_strategy["CRC"].sum())
print("Percentage of total patients with CRC detected:",  (df_selected_current_strategy["CRC"].sum() / df_test["CRC"].sum()).round(4) * 100, "%")

print("------------------------")
print("Selecting the same number of patients for the new strategy, that is, ", df_utilities_2016.sort_values(by = "FIT", ascending = False)[:tot_num_patients].shape[0])
print("Number of patentes with CRC by new strategy is: ", df_utilities_2016.sort_values(by = "FIT", ascending = False)[:tot_num_patients]["CRC"].sum())
print("Percentage of total patients with CRC detected:", (df_utilities_2016.sort_values(by = "FIT", ascending = False)[:tot_num_patients]["CRC"].sum() / df_test["CRC"].sum()).round(4) * 100, "%")

print("------------------------")
print("This is an improvement of ", df_utilities_2016.sort_values(by = "FIT", ascending = False)[:tot_num_patients]["CRC"].sum() - df_selected_current_strategy["CRC"].sum(), "patients")
print("A proportional improvement of ", ((df_utilities_2016.sort_values(by = "FIT", ascending = False)[:tot_num_patients]["CRC"].sum() - df_selected_current_strategy["CRC"].sum()) / df_selected_current_strategy["CRC"].sum()).round(4) * 100, "%")