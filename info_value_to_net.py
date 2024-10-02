import pysmile
import pysmile_license
import numpy as np
import pandas as pd

def info_value_to_net( val_data, net, output_dir = ""): 

    '''net.set_node_definition("Value_of_CRC_detection_by_screening", val_scr_data.values.transpose().flatten())
    net.set_node_definition("Value_of_CRC_detection_by_colonoscopy", val_col_data.values.transpose().flatten())'''

    net.set_node_definition("INFO", val_data.values.transpose().flatten())

    return net