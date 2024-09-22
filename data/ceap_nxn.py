import numpy as np
import pandas as pd
'''
      D_NA  D_C0  D_C1  D_C2  D_C3  D_C4  D_C5  D_C6
G_NA     0     0     0     2    18     5     0    22
G_C0     0     0     0     1     8     2     2    14
G_C1     0     0     1     0     3     2     1     0
G_C2     3     1     1    10    14     3     3    18
G_C3    20     9     0    15    80    16     3    14
G_C4     9     2     1     3    17    23     7     5
G_C5     7     5     1     5     4     2     6     8
G_C6    39    29     3    33     9     5     7    21
'''
# Index = Left member
# Columns = Right member
data = {
    'NA': [0, 0, 0, 3, 20, 9, 7, 39],
    'C0': [0, 0, 0, 1, 9, 2, 5, 29],
    'C1': [0, 0, 1, 1, 0, 1, 1, 3],
    'C2': [2, 1, 0, 10, 15, 3, 5, 33],
    'C3': [18, 8, 3, 14, 80, 17, 4, 9],
    'C4': [5, 2, 2, 3, 16, 23, 2, 5],
    'C5': [0, 2, 1, 3, 3, 7, 6, 7],
    'C6': [22, 14, 0, 18, 14, 5, 8, 21]
}
df = pd.DataFrame(data, index=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])

# Option 1
df.index = ['G_' + idx for idx in df.index]
df.columns = ['D_' + col for col in df.columns]

# Option 2 : Extract a subset DataFrame
subset_df = df.loc['G_C3':'G_C6', 'D_C3':'D_C6']

# Option 3 : numpy
arra = df.to_numpy()

# Print
print("Original DataFrame:")
print(df)
print("\nSubset DataFrame:")
print(subset_df)