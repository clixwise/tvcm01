import os
import pandas as pd
import os

def scri():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)

def file_oupu(file_path, text):
    with open(file_path, 'w') as f:
        f.write(text)
        
def file_inpu(file_path, deli=','):
    df = pd.read_csv(file_path, delimiter=',', header=0)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

if __name__ == "__main__":
    pass