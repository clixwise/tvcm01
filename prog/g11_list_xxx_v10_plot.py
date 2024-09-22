
from datetime import datetime
import inspect
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----
# Util
# ----
def get_calling_module_name():
    # Get the current frame and the frame of the caller
    caller_frame = inspect.stack()[2]
    # Get the module of the caller
    caller_module = inspect.getmodule(caller_frame[0])
    # Get the file name of the caller module
    caller_file = caller_module.__file__
    # Get the base name of the file (without the directory path)
    caller_file_name = os.path.basename(caller_file)
    print(f"Called by script: {caller_file_name}")
    return caller_file_name

# ----
# Plot
# ----
def dist_2xm_plot_ceap(what, df, desc):
    if False:
        data = {
            'obsM': [24, 5, 41, 126, 52, 29, 113],
            'obsF': [43, 6, 57, 123, 45, 24, 777]  # Modified to include an outlier
        }
        df = pd.DataFrame(data, index=pd.Index(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap'))

    # Inpu
    #:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
    #   '90-99'],
    #  dtype='object', name='age_bin') Index(['obsM', 'obsF'], dtype='object')
    print(f"df:{type(df)}\n{df}\n:{df.index} {df.columns}")
    colM = df.iloc[:, 0]
    colF = df.iloc[:, 1]
    colM_name = df.columns[0]
    colF_name = df.columns[1]

    # Get the current date and time
    date_work = datetime.now().strftime('%y-%m-%d %H:%M:%S')

    # Creating the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the data in the first subplot
    axs[0].plot(df.index, colM, label=colM_name, marker='o')
    axs[0].plot(df.index, colF, label=colF_name, marker='x')
    column_names = ','.join(df.columns)  # Dynamically get the column names and join them with ' and '
    axs[0].set_title(f'Fréquence de {desc[0]}=f({desc[1]}) - [{column_names}] {date_work}', fontsize=14)
    axs[0].set_ylabel('Fréquence', fontsize=12)
    axs[0].set_xlabel(f'{df.index.name}', fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)
    
    # Removing unused subplots
    axs[1].axis('off')
    
    plt.tight_layout()
    # plt.show()
    script_dire = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_name = get_calling_module_name()
    txt_file_name = f"{script_name} {what}.pdf"
    output_path = os.path.join(f'{script_dire}/../plot/', txt_file_name)
    plt.savefig(output_path) 

def dist_2xm_plot_agbi(what, df, desc):
    if False:
        data = {
            'obsM': [24, 5, 41, 126, 52, 29, 113],
            'obsF': [43, 6, 57, 123, 45, 24, 777]  # Modified to include an outlier
        }
        df = pd.DataFrame(data, index=pd.Index(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap'))

    # Inpu
    #:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
    #   '90-99'],
    #  dtype='object', name='age_bin') Index(['obsM', 'obsF'], dtype='object')
    print(f"df:{type(df)}\n{df}\n:{df.index} {df.columns}")
    colM = df.iloc[:, 0]
    colF = df.iloc[:, 1]
    colM_name = df.columns[0]
    colF_name = df.columns[1]

    # Get the current date and time
    date_work = datetime.now().strftime('%y-%m-%d %H:%M:%S')

    # Creating the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Plotting the data in the first subplot
    axs[0,0].plot(df.index, colM, label=colM_name, marker='o')
    axs[0,0].plot(df.index, colF, label=colF_name, marker='x')
    column_names = ','.join(df.columns)  # Dynamically get the column names and join them with ' and '
    axs[0,0].set_title(f'Fréquence de {desc[0]}=f({desc[1]}) - [{column_names}] {date_work}', fontsize=14)
    axs[0,0].set_ylabel('Fréquence', fontsize=12)
    axs[0,0].set_xlabel(f'{df.index.name}', fontsize=12)
    axs[0,0].legend(fontsize=10)
    axs[0,0].grid(True)

    # Dynamically creating the age_bins dictionary
    age_bins = {}
    for age_bin in df.index:
        start, end = map(int, age_bin.split('-'))
        age_bins[age_bin] = (start, end)
    # Prepare empty lists to hold continuous age data
    age_continuous = []
    gender_continuous = []

    # Generate random ages for Males
    for age_bin, freq in zip(df.index, df[colM_name]):
        age_min, age_max = age_bins[age_bin]
        ages = np.random.randint(age_min, age_max + 1, size=freq)  # Random ages in the bin
        age_continuous.extend(ages)
        gender_continuous.extend([colM_name] * freq)

    # Generate random ages for Females
    for age_bin, freq in zip(df.index, df[colF_name]):
        age_min, age_max = age_bins[age_bin]
        ages = np.random.randint(age_min, age_max + 1, size=freq)  # Random ages in the bin
        age_continuous.extend(ages)
        gender_continuous.extend([colF_name] * freq)

    # Create a DataFrame for the continuous distribution
    continuous_df = pd.DataFrame({
        f'{desc[1]}': age_continuous,
        f'{desc[0]}': gender_continuous
    })

    # Plotting the boxplot in the second subplot
    sns.boxplot(x=desc[0], y=desc[1], data=continuous_df, ax=axs[1,0])
    #sns.boxplot(data=df[continuous_df.columns], ax=axs[1,0])
    column_names = ','.join(continuous_df.columns)  # Dynamically get the column names and join them with ' and '
    axs[1,0].set_title(f'Boxplot de {desc[1]}=f({desc[0]}) - {date_work}', fontsize=14)
    axs[1,0].set_ylabel('Distribution', fontsize=12)
    axs[1,0].set_xlabel('Séries', fontsize=12)
    axs[1,0].set_xticklabels(df.columns)  # Set the x-axis tick labels to the column names from the DataFrame
    axs[1,0].legend(fontsize=10)
    axs[1,0].grid(True)
    
    # Setting up the violin plot
    #sns.violinplot(data=continuous_df, ax=axs[1,1])  # Placing it in the bottom-right corner
    sns.violinplot(x=desc[0], y=desc[1], data=continuous_df, ax=axs[1,1])
    axs[1,1].set_title(f'Violin plot de {desc[1]}=f({desc[0]}) - {date_work}', fontsize=14)
    axs[1,1].set_ylabel('Distribution')
    axs[1,1].set_xlabel('Séries', fontsize=12)
    axs[1,1].set_xticklabels(df.columns)
    axs[1,1].legend(fontsize=10)
    axs[1,1].grid(True)

    # Removing unused subplots
    axs[0, 1].axis('off')

    # Adjust the horizontal space between subplots
    plt.subplots_adjust(wspace=0.1) # plt.subplots_adjust(hspace=0.4)  
    plt.tight_layout()
    # plt.show()
    script_dire = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_name = get_calling_module_name()
    txt_file_name = f"{script_name} {what}.pdf"
    output_path = os.path.join(f'{script_dire}/../plot/', txt_file_name)
    plt.savefig(output_path) 

def dist_nxm_plot(what, df, desc):

    if False:
        data = {
            'C0': [0, 0, 1, 9, 2, 5, 29],
            'C1': [0, 1, 1, 0, 1, 1, 3],
            'C2': [1, 0, 10, 15, 3, 5, 33],
            'C3': [8, 3, 14, 80, 17, 4, 9],
            'C4': [2, 2, 3, 16, 23, 2, 5],
            'C5': [2, 1, 3, 3, 7, 6, 7],
            'C6': [14, 0, 18, 14, 5, 8, 21]
        }
        df = pd.DataFrame(data, index=pd.Index(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap_L'))

    # Get the current date and time
    date_work = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    print (df)
    df = df.iloc[::-1]
    # Creating the heatmap
    plt.figure(figsize=(10, 8))
    cmap='Purples'
    annot = np.where(df != 0, df.astype(str) + '%', "")
    titl = f'Insuffisance veineuse des membre'
    ylab = 'Proportion (%)'

    fig, axs = plt.subplots(figsize=(14, 8))
    mask = np.ma.masked_equal(df, 0).mask 
    heatmap = sns.heatmap(df, mask=mask, annot=annot, fmt='', cmap=cmap, cbar=True, linewidths=0.7, linecolor='lightgray', annot_kws={"size": 12})

    # Setting the title and labels
    axs.set_title(f'Heatmap de {desc[1]}=f({desc[0]}) - {date_work}', fontsize=14)
    axs.set_xlabel(f'{desc[0]}', fontsize=14)
    axs.set_ylabel(f'{desc[1]}', fontsize=14)

    plt.tight_layout()
    # plt.show()
    script_dire = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_name = get_calling_module_name()
    txt_file_name = f"{script_name} {what}.pdf"
    output_path = os.path.join(f'{script_dire}/resu/', txt_file_name)
    plt.savefig(output_path) 
    
# Example usage
if __name__ == "__main__":
    pass
