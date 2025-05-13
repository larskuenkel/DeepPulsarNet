import os
import pandas as pd

cwd = os.getcwd()

def replace_paths_in_df(df, path):
    for (index, row) in df.iterrows():
        old_path = row['FileName']
        new_path = path + old_path.split('/')[-1]
        df.at[index, 'FileName'] = new_path
    return df


if os.path.exists("./pmps_sample_set/"):
    print("Sample set is located in tutorial folder.")
else:
    print("Sample set needs to be downloaded.")
    os.system('wget -O pmps_obs_sample_v3.tar.gz -c https://zenodo.org/records/15399789/files/pmps_obs_sample_v3.tar.gz')
    os.system('tar -xvf pmps_obs_sample_v3.tar.gz')

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

df_train.drop(list(df_train.filter(regex = 'Unnamed')), axis = 1, inplace = True)
df_test.drop(list(df_test.filter(regex = 'Unnamed')), axis = 1, inplace = True)

df_train.to_csv('../deeppulsarnet/datasets/pmps_sample_train.csv')
df_test.to_csv('../deeppulsarnet/datasets/pmps_sample_test.csv')

print('Modified csvs to ../deeppulsarnet/datasets/.')

print(f"Created: ../deeppulsarnet/datasets/pmps_sample_train.csv")
print(f"To use the set use the option: --path_noise pmps_sample_train.csv")

print(f"The test set can be analysed after training a model with 'python create_model_prediction.py -m test_model.pt -f pmps_sample_test.csv'. ")