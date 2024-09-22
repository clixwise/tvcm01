import pandas as pd
'''
ke39 : ceap_mbre:<class 'pandas.core.frame.DataFrame'>
      G    D
NA   39   66
C0   24   43
C1    5    6
C2   41   57
C3  126  123
C4   52   45
C5   29   24
C6  113   83
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['G', 'D'], dtype='object')
sum:876
'''
data = {
    'G': [39, 24, 5, 41, 126, 52, 29, 113],
    'D': [66, 43, 6, 57, 123, 45, 24, 83]
}
df = pd.DataFrame(data, index=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])

# Print the DataFrame
print("df:")
print(df)

# Calculate the sum of all elements in the DataFrame
df_sum = df.sum().sum()

# Print the sum
print("\ndf.sum():")
print(df_sum)

# Restructure cols as lists to be used by stat tests
df_stat = df.copy()
df_stat.columns = ['obsM', 'obsF']
obsM = df_stat['obsM'].tolist()
obsF = df_stat['obsF'].tolist()