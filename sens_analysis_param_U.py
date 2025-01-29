
from elicitation import parameter_elicitation_utilities_linear
import numpy as np
import matplotlib.pyplot as plt



def sens_analysis_param_U(net, info_array, cost_array, PE_array):

    # Prepare a grid to store the results (only 2D slices for visualization)
    max_function_idx = np.zeros((len(info_array), len(cost_array)))
    max_function_value = np.zeros((len(info_array), len(cost_array)))

    # Loop through param1 and param2 (for a fixed param3)
    # fixed_param3 = 0.7  # You can vary this to explore different slices

    possible_outcomes = net.get_outcome_ids("Screening")
    function_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'magenta']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    rho_comfort = net.get_node_value("Value_of_comfort")[2]

    for k, param3 in enumerate(PE_array):
        max_function_idx = np.zeros((len(info_array), len(cost_array)))
        max_function_value = np.zeros((len(info_array), len(cost_array)))

        for i, param1 in enumerate(info_array):
            for j, param2 in enumerate(cost_array):
                # Call the custom function with the current combination of parameters
                params = parameter_elicitation_utilities_linear(net, PE = param3, PE_info = param1, PE_cost = param2, rho_comfort = rho_comfort, value_function = "rel_point_cond_mut_info", logging = None)
            
                net.set_mau_expressions(node_id = "U", expressions = [f"{params[0]} - {params[1]}*Exp( - {params[2]} * V)"])
                net.update_beliefs()

                results = net.get_node_value("Screening")
                
                max_idx = np.argmax(results)
                max_val = results[max_idx]
                
                # Determine which function has the maximum value
                max_function_idx[i,j] = max_idx
                max_function_value[i,j] = max_val

        # Plotting the results (for the slice at fixed param3)
        im = axes[k].imshow(max_function_idx.T, cmap='tab10', origin='lower',
                extent=(info_array.min(), info_array.max(), 
                        cost_array.min(), cost_array.max()),
                        alpha = 0)


        for i, param1 in enumerate(info_array):
            for j, param2 in enumerate(cost_array):
                max_idx = int(max_function_idx[i, j])  # Convert to int for index
                label = possible_outcomes[max_idx]       # Get corresponding label
                value = max_function_value[i, j]       # Get corresponding value
                color = function_colors[max_idx]  
                
                # Annotate the plot with function label and value (centered text)
                axes[k].text(param1, param2, f'{label}\n{value:.2f}', 
                        ha='center', va='center', color='black',
                    bbox=dict(facecolor=color, alpha=0.5, edgecolor='black'))
            
        axes[k].set_title(f'Function with Maximum Value (Fixed Param3={param3})')
        axes[k].set_xlabel('Param1')
        axes[k].set_ylabel('Param2')
        axes[k].set_xlim(info_array.min() - 0.5, info_array.max() + 0.5)
        axes[k].set_ylim(cost_array.min() - 50, cost_array.max() + 50)
        axes[k].set_xticks(info_array)
        axes[k].set_yticks(cost_array)

        # Force a square aspect ratio by adjusting the plot limits
        axes[k].set_aspect(abs((info_array.max() - info_array.min()) / 
                            (cost_array.max() - cost_array.min())))
        

    # fig.colorbar(im, ax=axes.ravel().tolist(), ticks=np.arange(7), label='Function Index')

    plt.tight_layout()
    plt.show()

    plt.savefig("outputs/sens_analysis_param_U.png")

    return