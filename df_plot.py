import pandas as pd
import numpy as np

def plot_df(df, net, outcomes, mod = "cmi"):

    # Get instances from variable
    res_scr_outcomes = net.get_outcome_ids(outcomes[0])
    crc_outcomes = net.get_outcome_ids(outcomes[1])
    
    new_data = df.reshape(-1, 3).transpose()

    # Create a multi-index from tuples
    if mod == "cmi":
        scr_outcomes = net.get_outcome_ids(outcomes[2])
        index = pd.MultiIndex.from_tuples([(i, j) for i in crc_outcomes for j in scr_outcomes], names=[outcomes[1], outcomes[2]])
    elif mod == "mi":
        index = crc_outcomes
    # Create a DataFrame with the multi-index
    df_new = pd.DataFrame(new_data, index=res_scr_outcomes, columns=index)

    return df_new