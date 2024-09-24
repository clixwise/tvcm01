import os
import sys
import pandas as pd
'''
ceap     NA  C0  C1  C2  C3  C4  C5  C6
age_bin
10-19     3   1   0   1   5   0   1   1
20-29     5   2   1   5   8   3   1   6
30-39     8   9   0  15  21   8   4  22
40-49    24   7   2   9  38  16   7  32
50-59    25  17   3  22  66  12  12  50
60-69    15  16   4  22  54  38  15  33
70-79    17  11   1  18  46  18  11  40
80-89     8   4   0   6   9   2   2  11
90-99     0   0   0   0   2   0   0   1
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
       '90-99'],
      dtype='object', name='age_bin')
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object', name='ceap') sum:876

'''
if __name__ == "__main__":

      exit_code = 0
            
      script_path = os.path.abspath(__file__)
      script_dir = os.path.dirname(script_path)
      script_name = os.path.basename(__file__)

      print (f"len(sys.argv): {len(sys.argv)}")
      print (f"sys.argv: {sys.argv}")
      if len(sys.argv) == 2:
            file_path = sys.argv[1]
      else:
            file_path = script_dir

      # ----
      # 1:Inpu
      # ----
      file_inpu = "../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
      path_inpu = os.path.join(file_path, file_inpu)
      df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False)
      #
      agbi_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
      ceap_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
      #
      df2 = pd.pivot_table(df1, index='age_bin', columns='ceap', aggfunc='size', fill_value=0)
      df2 = df2.reindex(index=agbi_list, columns=ceap_list, fill_value=0)
      df2 = df2.rename_axis('age_bin', axis='index')
      print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
      
      # ----
      # 2:Chck with row & columnn 'tota'
      # ----
      df2_sum = df2.sum().sum()
      df2['tota'] = df2.sum(axis=1)
      # Calculate total sum across all columns and add as a new row
      total_row = df2.sum(axis=0)
      total_row.name = 'tota'  # Set the name for the total row
      df2 = pd.concat([df2, total_row.to_frame().T])
      df2 = df2.rename_axis('age_bin', axis='index')
      print(f"\nStep 2 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns} sum:{df2_sum}")

      # ----
      # 3: Drop row & columnn 'tota'
      # ----
      df2 = df2.drop(index='tota')
      df2 = df2.drop(columns='tota')
      df2 = df2.rename_axis('age_bin', axis='index')
      print(f"\nStep 3 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns} sum:{df2_sum}")
