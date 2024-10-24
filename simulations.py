import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pdb
from matplotlib.patches import FancyBboxPatch

def simulate_test_results(sensitivity_scr, specificity_scr, y_crc):
    """
    Simulate test results based on sensitivity, specificity, and actual number of patients
    with and without the disease.
    
    Parameters:
    - sensitivity (float): Sensitivity of the test (true positive rate)
    - specificity (float): Specificity of the test (true negative rate)
    - num_with_disease (int): Number of patients who have the disease
    - num_without_disease (int): Number of patients who do not have the disease
    
    Returns:
    - pandas DataFrame: A DataFrame with the simulated test results, true conditions, and test outcomes.
    """
    

    num_with_disease = y_crc.sum()
    num_without_disease = len(y_crc) - num_with_disease

    # Step 1: Create a list of patients with and without the disease
    
    # Step 2: Simulate test results
    scr_results = []
    col_results = []
    
    for y in y_crc:
        if y == 1:
            # Patient has the disease, test is positive with probability = sensitivity
            scr_result = np.random.choice([1, 0], p=[sensitivity_scr, 1 - sensitivity_scr])
        else:
            # Patient does not have the disease, test is negative with probability = specificity
            scr_result = np.random.choice([0, 1], p=[specificity_scr, 1 - specificity_scr])

        scr_results.append(scr_result)
    
    # Step 3: Create a DataFrame to store the results
    df_scr = pd.DataFrame({
        'Condition': y_crc,    # True condition of the patient
        'TestResult': scr_results  # Simulated test result
    })

    
    
    return df_scr



def plot_classification_results(y_true=None, y_pred=None, report_df = None, conf_matrix = None, std_conf_matrix = None, total_cost= None, label = "", plot = True, log_dir = None):

    if report_df is None:

        # Create a classification report
        report = classification_report(y_true, y_pred, output_dict=True)

        # Convert the classification report into a DataFrame for easier visualization
        report_df = pd.DataFrame(report).transpose()

        # Generate a confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate sensitivity (recall) and specificity manually
        tn, fp, fn, tp = conf_matrix.ravel()

        # Sensitivity (Recall) is already included in classification report
        sensitivity = report['True']['recall']  # For class 1 (positive class)

        # Specificity calculation
        specificity = tn / (tn + fp)

        # Add Sensitivity and Specificity to the DataFrame
        report_df.loc['sensitivity'] = [sensitivity, np.nan, np.nan, np.nan]
        report_df.loc['specificity'] = [specificity, np.nan, np.nan, np.nan]


    if plot:
        # Plot the confusion matrix using Seaborn for a heatmap
        fig, ax = plt.subplots(1,2, figsize=(12, 5))

        if std_conf_matrix is not None:
            annotations = np.array([f"{mean:.2f}±{std:.2f}" for mean, std in zip(conf_matrix.flatten(), std_conf_matrix.flatten())])
            annotations = annotations.reshape(conf_matrix.shape)
            fmt = ''
        else:
            annotations = True
            fmt = '.1f'

        # First subplot: Confusion matrix
        sns.heatmap(conf_matrix, annot=annotations, fmt=fmt, cmap='Blues', cbar=False, annot_kws={"size": 14}, ax=ax[0])
        ax[0].set_title('Confusion Matrix')
        ax[0].set_ylabel('True label')
        ax[0].set_xlabel('Predicted label')

        # Second subplot: Classification report as a heatmap
        sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', cbar=False, fmt='.2f', ax=ax[1])
        ax[1].xaxis.tick_top()
        ax[1].xaxis.set_label_position('top')
        ax[1].set_title('Classification Metrics')
        ax[1].set_ylabel('Metrics')
        # ax[1].set_xlabel('Classes')

        ax[1].text(0.39, 0.15, r"Total cost of the strategy: $\bf{" + f"{total_cost:,.2f}" + "€}$", color='black', fontsize=9,
            ha='left', va='center', transform=ax[1].transAxes)
        ax[1].text(0.39, 0.10, r"Mean cost per patient:  $\bf{" + f"{total_cost/(conf_matrix.sum()):,.2f}" + "€}$", color='black', fontsize=9,
            ha='left', va='center', transform=ax[1].transAxes)
        box = FancyBboxPatch((0.47, 0.12), 0.42, 0.03, boxstyle="round,pad=0.1", 
                     linewidth=2, edgecolor='black', facecolor='none', transform=ax[1].transAxes)
        ax[1].add_patch(box)

        plt.tight_layout()
        plt.savefig(f"{log_dir}/{label}_classification_results.png", dpi=200)
        plt.close(fig)

    return report_df, conf_matrix







def output_test_results_extended(df_scr, cost_scr, df_col, cost_col, y_crc, verbose = False):

    # Add columns to indicate true positives, false positives, etc.
    df_scr['TruePositive'] = (df_scr['Condition'] == 1) & (df_scr['TestResult'] == 1)
    df_scr['FalsePositive'] = (df_scr['Condition'] == 0) & (df_scr['TestResult'] == 1)
    df_scr['TrueNegative'] = (df_scr['Condition'] == 0) & (df_scr['TestResult'] == 0)
    df_scr['FalseNegative'] = (df_scr['Condition'] == 1) & (df_scr['TestResult'] == 0)
    
    # Step 4: Calculate confusion matrix components
    TP_scr = df_scr['TruePositive'].sum()
    FP_scr = df_scr['FalsePositive'].sum()
    TN_scr = df_scr['TrueNegative'].sum()
    FN_scr = df_scr['FalseNegative'].sum()
    
    # Create confusion matrix
    confusion_matrix_scr = pd.DataFrame({
        'Predicted Negative': [TN_scr, FN_scr],
        'Predicted Positive': [FP_scr, TP_scr]
    }, index=['Actual Negative', 'Actual Positive'])


    FIT_positives = df_scr[df_scr["TestResult"] == 1]
    patient_data = df_scr["Condition"]

    if verbose:
        print("Number of patients considered: ", patient_data.shape[0])
        print(f"Cost of screening: {cost_scr*(patient_data.shape[0])} €")
        print("Number of FIT positives: ", FIT_positives.shape[0])
        print("Number of colonoscopies to be done: ", FIT_positives.shape[0])
        print(f"Cost of colonoscopy program: {cost_col*FIT_positives.shape[0]} €")

    
    # Add columns to indicate true positives, false positives, etc.
    df_col['TruePositive'] = (df_col['Condition'] == 1) & (df_col['TestResult'] == 1)
    df_col['FalsePositive'] = (df_col['Condition'] == 0) & (df_col['TestResult'] == 1)
    df_col['TrueNegative'] = (df_col['Condition'] == 0) & (df_col['TestResult'] == 0)
    df_col['FalseNegative'] = (df_col['Condition'] == 1) & (df_col['TestResult'] == 0)

    # Step 6: Calculate confusion matrix components
    TP_col = df_col['TruePositive'].sum()
    FP_col = df_col['FalsePositive'].sum()
    TN_col = df_col['TrueNegative'].sum()
    FN_col = df_col['FalseNegative'].sum()

    # Create confusion matrix
    confusion_matrix_col = pd.DataFrame({
        'Predicted Negative': [TN_col, FN_col],
        'Predicted Positive': [FP_col, TP_col]
    }, index=['Actual Negative', 'Actual Positive'])

    total_cost = cost_scr*df_scr["Condition"].shape[0] + cost_col*FIT_positives.shape[0]

    if verbose:
        print("Number of CRC true positive cases detected by colonoscopy: ", TP_scr)
        print("Number of false positives by colonoscopy: ", FP_scr)
        print(f"Total cost of screening and colonoscopy: {total_cost} €")
        print("Proportion of total CRC cases in the whole population detected by the method: ", TP_scr / y_crc.sum())
        # print("Proportion of cases in the high-risk target population detected by the method: ", TP_scr / y.sum())

    combined_confusion_matrix = pd.DataFrame({
        'Predicted Negative': [TN_scr + TN_col, FN_scr + FN_col],
        'Predicted Positive': [FP_col, TP_col]
    }, index=['Actual Negative', 'Actual Positive'])

    # Calculate sensitivity and specificity using the combined confusion matrix
    sensitivity = TP_col / (TP_col + FN_col + FN_scr)
    specificity = (TN_scr + TN_col) / (TN_scr +TN_col + FP_col)
    PPV = TP_col / (TP_col + FP_col)
    NPV = (TN_scr + TN_col) / (TN_scr + TN_col + FN_scr + FN_col)

    metrics = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": PPV,
        "NPV": NPV
    }
    
    return confusion_matrix_scr, confusion_matrix_col, combined_confusion_matrix, total_cost, metrics
    



