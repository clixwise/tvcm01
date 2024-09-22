import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': ['apple', 'banana'],
    'B': ['cat', 'dog']
})

# DataFrame name
df_name = 'df'

# Adjusting all cells by prefixing with the DataFrame name
for col in df.columns:
    df[col] = df[col].apply(lambda x: f"{df_name[-1]}_{x}")

print(df)

# Sample DataFrames with string content
dfsource1 = pd.DataFrame({'A': ['apple', '', 'banana'], 'B': ['cat', 'dog', '']}, index=[0, 1, 2])
dfsource2 = pd.DataFrame({'A': ['orange', '', 'grape'], 'B': ['fish', 'hamster', '']}, index=[0, 1, 2])

# Sample target DataFrame with existing data
dftarget = pd.DataFrame({'A': ['existing1', 'existing2', ''], 'B': ['existing3', '', 'existing5']}, index=[0, 1, 2])

# Merging cell by cell into dftarget
for i in range(dftarget.shape[0]):  # Iterate over rows
    for j in range(dftarget.shape[1]):  # Iterate over columns
        existing_value = dftarget.iat[i, j]  # Get existing value
        new_value1 = dfsource1.iat[i, j]  # Get value from dfsource1
        new_value2 = dfsource2.iat[i, j]  # Get value from dfsource2
        
        # Create a list to hold non-empty values
        values_to_merge = [existing_value, new_value1, new_value2]
        
        # Filter out empty strings and join the remaining values with a space
        merged_value = ' '.join(filter(bool, values_to_merge))
        
        # Assign the merged value back to the target DataFrame
        dftarget.iat[i, j] = merged_value

print(dftarget)


# Create the DataFrame with the given data
data = [['--', '-', '', '', '', '', '', '++'],
        ['-', '', '', '', '', '', '', '++'],
        ['', '', '++', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '++'],
        ['', '', '', '', '++', '', '-', '--'],
        ['', '', '', '', '', '++', '', '--'],
        ['', '', '', '', '', '', '++', ''],
        ['++', '++', '', '+', '--', '-', '', '']]

columns = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
index = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

df = pd.DataFrame(data, columns=columns, index=index)

# Custom colormap for "L" (light blue), "++" (dark blue), "R" (light orange), "RR" (dark orange)
cmap = ListedColormap(['white', 'blue', 'blue', 'orange', 'orange'])  # White for space, blue, and orange shades

# Map the DataFrame to numeric values for the heatmap
value_map = {'': 0, '+': 1, '++': 2, '-': 3, '--': 4}
df_mapped = df.replace(value_map)

# Plot the heatmap with inverted Y-axis
plt.figure(figsize=(8, 6))
if True:
    sns.heatmap(df_mapped,
                annot=df,
                cmap=cmap, cbar=False, linewidths=1, linecolor='black', fmt='', 
                annot_kws={"size": 16}, square=True)
else: # Inverted Y axis
    sns.heatmap(df_mapped.iloc[::-1],  # Invert Y-axis
                annot=df.iloc[::-1],  # Invert annotations
                cmap=cmap, cbar=False, linewidths=1, linecolor='black', fmt='', 
                annot_kws={"size": 16}, square=True)

# Add labels and title
plt.xlabel("Membre droit", fontsize=14)
plt.ylabel("Membre gauche", fontsize=14)
plt.title("Heatmap Residuals", fontsize=16)

plt.show()

pass