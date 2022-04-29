import os
import pandas as pd

cwd = os.getcwd()

def replace_paths_in_df(df, path):
    for (index, row) in df.iterrows():
        old_path = row['FileName']
        new_path = cwd + old_path.split('/')[-1]
        df.at[index, 'FileName'] = new_path
    return df



if os.path.exists("./pmps_sample_set/"):
    print("Sample set is located in tutorial folder.")
else:
    print("Sample set needs to be downloaded.")
    os.system('wget -O pmps_obs_sample_v2.tar.gz -c https://uni-bielefeld.sciebo.de/s/LoENwCQgzV8VFMg/download')
    os.system('tar -xvf pmps_obs_sample_v2.tar.gz')

if os.path.exists("./pmps_sample_set/pmps_sample_train.csv"):
    print("Train Data csv is found.")
else:
    print("Train Data csv is found. The download failed for some reason.")
    os._exit(0)

df_train = pd.read_csv('./pmps_sample_set/pmps_sample_train.csv')
df_test = pd.read_csv('./pmps_sample_set/pmps_sample_test.csv')

df_train = replace_paths_in_df(df_train, cwd + '/pmps_sample_set/training/')
df_test = replace_paths_in_df(df_test, cwd + '/pmps_sample_set/test/')

print('Paths modified.')

df_train.to_csv('../deeppulsarnet/datasets/pmps_sample_train.csv')
df_test.to_csv('../deeppulsarnet/datasets/pmps_sample_test.csv')

print('Modified csvs to ../deeppulsarnet/datasets/.')