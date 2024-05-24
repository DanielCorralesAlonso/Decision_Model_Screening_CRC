import pysmile
import pysmile_license
import numpy as np
import pandas as pd
from plot_cond_mut_info import plot_cond_mut_info

np.seterr(divide='ignore', invalid = 'ignore')

from df_plot import plot_df 
from info_value_to_net import info_value_to_net
from get_info_values import conditional_mutual_info, pointwise_conditional_mutual_info

# Read the network
net = pysmile.Network()
net.read_file("genie_models/Basic_ID_screening_22052024.xdsl")

cond_mut_info_scr, cond_mut_info_col = conditional_mutual_info(net)

print(cond_mut_info_col)

point_cond_mut_info_scr, point_cond_mut_info_col = pointwise_conditional_mutual_info(net)

print(point_cond_mut_info_col)

point_cond_mut_info_scr, point_cond_mut_info_col = pointwise_conditional_mutual_info(net, normalize = True)

print(point_cond_mut_info_col)


plot_cond_mut_info(net)