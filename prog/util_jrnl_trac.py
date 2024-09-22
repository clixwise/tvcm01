from datetime import datetime
import inspect
import os
import pandas as pd

class Repo:
    def __init__(self):
        self.df_jrnl = pd.DataFrame(columns=['what'])
        self.dict = {}
        self.invo = {}
repo = Repo()
def plug_init():
    repo = Repo()
    return repo
def plug_fini(repo : Repo):
    collapsed_df = repo.df_jrnl.groupby('what').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    #print(collapsed_df)
    return collapsed_df
def plug_inpu(what, repo, df_inpu, df_desc, stac=1):
    caller_f_code = inspect.stack()[stac].frame.f_code
    colu_name = caller_f_code.co_name.replace('_test_', '_')
    print(f"'{what}'' df:{type(df_inpu)}\n{df_inpu}\n:{df_inpu.index}")
    plug_dict = {'inpu' : df_inpu, 'desc' : df_desc, 'oupu' : None}
    repo.invo[f"'{what}'"] = plug_dict
def plug_oupu(what, repo, df_oupu, stac=1):
    caller_f_code = inspect.stack()[stac].frame.f_code
    colu_name = caller_f_code.co_name.replace('_test_', '_')
    print(f"'{what}' '{colu_name}' df:{type(df_oupu)}\n{df_oupu}\n:{df_oupu.index}")
    plug_dict = repo.invo[f"'{what}'"]
    plug_dict['oupu'] = df_oupu
# not used :
def plug_exec(repo : Repo, what, mtrx_inpu, diff, mtrx_oupu, stac=1):
    caller_f_code = inspect.stack()[stac].frame.f_code
    colu_name = caller_f_code.co_name.replace('_test_', '_')
    if colu_name not in repo.df_jrnl.columns:
        repo.df_jrnl[colu_name] = None
    repo.df_jrnl.loc[what, 'what'] = what
    repo.df_jrnl.loc[what, colu_name] = diff
    #
    repo.dict[f'{what} : {colu_name}'.upper()] = [mtrx_inpu, mtrx_oupu]
def plug_head(repo : Repo, what, mtrx_inpu):
    #caller_f_code = inspect.stack()[1].frame.f_code
    #colu_name = caller_f_code.co_name.replace('main', '')
    key = f'{what}'.upper()
    val = f"{'-' * len(key)}"
    repo.dict[key] = [mtrx_inpu,val]
'''
Init
'''
def pgrm_init(): 
    repo = plug_init()
    return repo
'''
Fini
'''
def pgrm_fini(repo, script_name_full):
    print(f"++++++++++++++++")
    print(f"SYNTHESIS : INIT")
    print(f"++++++++++++++++")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    date_work = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    print (repo.invo)
    # ---
    script_dire = os.path.dirname(os.path.abspath(__file__))
    script_name = script_name_full[0]
    file_path = os.path.join(f'{script_dire}/../oupu/' f'{script_name}.info.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"{date_work}\n")
        for what, plug_dict in repo.invo.items():
            file.write(f"\n{what.upper()}\n")
            df_inpu = plug_dict['inpu']
            desc = plug_dict['desc']
            df_oupu = plug_dict['oupu']
            expl_colu_list = [col for col in ['expl_stat', 'expl_pval'] if col in df_oupu.columns]
            df_work = df_oupu.drop(columns=expl_colu_list)
            file.write(f'{df_inpu}\n-\n{desc}\n-\n{df_work}\n')
        if False:
            for what, plug_dict in repo.invo.items():
                df_oupu = plug_dict['oupu']
                expl_colu_list = [col for col in ['expl_stat', 'expl_pval'] if col in df_oupu.columns]
                if expl_colu_list:
                    df_work = df_oupu[['sign_stat','expl_stat','expl_pval']] 
                    file.write(f"-\n{df_work}")
            
    pd.reset_option('^display')
    print(f"++++++++++++++++")
    print(f"SYNTHESIS : FINI")
    print(f"++++++++++++++++")

if __name__ == "__main__":
    pass