
from functions import *

import yaml
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

import pdb
    

def parameter_elicitation_utilities_tanh(PE_info, PE_cost, rho_comfort):

    best_info = 1   # 0.601
    worst_info = 0      # 0.042 

    best_cost = 0       #12.14
    worst_cost = 8131.71    #1000

    w_best = tanh_fun(best_info, rho_comfort)
    w_worst = tanh_fun(worst_info, rho_comfort)
    w_PE = tanh_fun(PE_info, rho_comfort)

    v_best = ((8131.71 - best_cost) / 8131.71) * w_best
    v_worst = ((8131.71 - worst_cost) / 8131.71) * w_worst
    v_PE = ((8131.71 - PE_cost) / 8131.71) * w_PE

    print("Searching for a solution of the system of equations...")
    num_points = 1000

    # Generate a list of tuples with random initial points
    init_list = [tuple(np.random.uniform(1, 1, 3)) for _ in range(num_points)]

    params = None
    for init in init_list:
        try:
            params = system_of_eq(y = v_PE, p = cfg["PE_prob"] , init = init, min_value = v_worst, max_value = v_best)
        except:
            continue

    if params is None:
        print("No solution found...")
        return None

    return params


def parameter_elicitation_utilities_linear(net, PE, PE_info, PE_cost, rho_comfort, value_function, logging = None):
    net.update_beliefs()

    # pdb.set_trace()
    worst_info = min(net.get_node_value("INFO"))
    best_info = max(net.get_node_value("INFO"))

    best_cost = 0    
    worst_cost = 8131.71  

    v_best = rho_comfort * best_info - np.log10(best_cost+1)
    v_worst =  rho_comfort * worst_info - np.log10(worst_cost+1)
    v_PE =  rho_comfort * PE_info - np.log10(PE_cost+1)

    if logging is not None:
        logging.info(f"Best info: {best_info}")
        logging.info(f"Worst info: {worst_info}")
        logging.info(f"PE info: {PE_info}")

        logging.info(f"v_best: {v_best}")
        logging.info(f"v_worst: {v_worst}")
        logging.info(f"v_PE: {v_PE}")

        logging.info("Searching for a solution of the system of equations...")

    num_points = 500

    # Generate a list of tuples with random initial points
    if value_function == "point_cond_mut_info":
        init_list = [tuple(np.random.uniform(-0.01, 0.01, 3)) for _ in range(num_points)]
    elif value_function == "rel_point_cond_mut_info":
        init_list = [tuple(np.random.uniform(-1, 1, 3)) for _ in range(num_points)]

    params = None
    for init in init_list:
        try:
            params = system_of_eq(y = v_PE, p = PE , init = init, min_value = v_worst, max_value = v_best)
            break
        except:
            continue

    if params is None:
        if logging is not None:
            logging.warning("No solution found...")
        return None

    return params