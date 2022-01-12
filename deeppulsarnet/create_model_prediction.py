import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd
import argparse
import os


def main(obs_file, model):
    try:
        os.mkdir('./predictions')
    except FileExistsError:
        pass
    model_name = model.split('.pt')[0]
    csv = obs_file
    csv_name = csv.split('.')[0]
    net = torch.load(f'./trained_models/{model}')
    df = pd.read_csv(f'./datasets/{csv}')
    df['Pulsar Prediction'] = -1.
    for (index, row) in df.iterrows():
        file = row['FileName']
        im, clas, clas_single, (cands, cands_target) = net.test_single_file(file)
        softmaxed = F.softmax(clas[0, :2,:], dim=0)
        max_pred = softmaxed[1,:].max()
        df.at[index, 'Pulsar Prediction'] = max_pred
        if not cands.nelement() == 0:
            softmaxed_cands = F.softmax(cands[:, :2], dim=1)
            for i in range(cands.shape[0]):
                if not f'Cand {i} Pred' in df.columns:
                    df[f'Cand {i} Pred'] = -1.
                    df[f'Cand {i} P0'] = -1.
                df.at[index, f'Cand {i} Pred'] = softmaxed_cands[i,1]
                df.at[index, f'Cand {i} P0'] = cands[i,2]
        if index %1000 ==0:
            df.to_csv(f"./predictions/{csv_name}_{model_name}.csv")
    print(df)
    df.to_csv(f"./predictions/{csv_name}_{model_name}.csv")
    print(f"{os.getcwd()}/predictions/{csv_name}_{model_name}.csv created")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grab parameters.')
    parser.add_argument('-f', type=str, help='Obs file')
    parser.add_argument('-m', type=str, help='Model')

    args = parser.parse_args()
    df = main(args.f, args.m)
