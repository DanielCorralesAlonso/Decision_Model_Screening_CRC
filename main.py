import pysmile
import pysmile_license
import numpy as np
import pandas as pd
from plots import plot_cond_mut_info
from save_info_values import save_info_values
np.seterr(divide='ignore', invalid = 'ignore')

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import mutual_info_measures


# Read the network
print("Reading network...")
net = pysmile.Network()
net.read_file("genie_models/Basic_ID_screening_current.xdsl")


print("Calculating pointwise conditional mutual information values...")
df_value_scr, df_value_col = save_info_values(net)
net = info_value_to_net(df_value_scr, df_value_col, net)


print("Saving network...")
net.write_file("genie_models/Basic_ID_screening_current.xdsl")


print("Plotting value functions...")
plot_cond_mut_info(net)


print("Done!")