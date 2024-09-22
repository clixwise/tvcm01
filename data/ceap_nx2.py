import numpy as np
import pandas as pd
'''
{
'NA': array([[257,  66], [ 39,   0]]), 
'C0': array([[295,  43], [ 24,   0]]), 
'C1': array([[352,   5], [  4,   1]]), 
'C2': array([[274,  47], [ 31,  10]]), 
'C3': array([[193,  43], [ 46,  80]]), 
'C4': array([[288,  22], [ 29,  23]]), 
'C5': array([[315,  18], [ 23,   6]]), 
'C6': array([[187,  62], [ 92,  21]])
}
'''
data = {
    'cont_tabl': [
        [[257, 66], [39, 0]],
        [[295, 43], [24, 0]],
        [[352, 5], [4, 1]],
        [[274, 47], [31, 10]],
        [[193, 43], [46, 80]],
        [[288, 22], [29, 23]],
        [[315, 18], [23, 6]],
        [[187, 62], [92, 21]]
    ]
}
df = pd.DataFrame(data, index=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])

# Print the DataFrame
print(df)

nump = {indx: np.array(item) for indx, item in df['cont_tabl'].items()}
print("\nNumpy:")
print(nump)