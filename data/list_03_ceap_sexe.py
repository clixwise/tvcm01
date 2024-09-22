import pandas as pd
'''
ke37 : ceap_sexe:<class 'pandas.core.frame.DataFrame'>
     M    F
NA  52   53
C0  31   36
C1   5    6
C2  44   54
C3  93  156
C4  38   59
C5  18   35
C6  97   99
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['M', 'F'], dtype='object')
df2.sum:876
'''
data = {
    'M': [52, 31, 5, 44, 93, 38, 18, 97],
    'F': [53, 36, 6, 54, 156, 59, 35, 99]
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