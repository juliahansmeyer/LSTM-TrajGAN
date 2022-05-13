"""Convert encoded csv files to one-hot-encoded npy files."""

import pandas as pd
import numpy as np
import argparse

def data_conversion(df, tid_col):
    """Converts input panda dataframe to one-hot-encoded Numpy array (locations are still in float).
    
    Returns array of length 5 (lat-lon, day, hour, category,mask)
    Each array has 2052 values, because there are 2052 trajectories
    """
    

    x = [[] for i in ['lat_lon', 'day', 'hour', 'category', 'mask']]
    for tid in df[tid_col].unique():
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, ['lat', 'lon', 'day', 'hour', 'category']].values)
        loc_list = []
        for i in range(0, len(traj)):
            lat = traj['lat'].values[i]
            lon = traj['lon'].values[i]
            loc_list.append(np.array([lat, lon], dtype=np.float64))
        x[0].append(loc_list)
        x[1].append(np.eye(7)[features[2].astype(np.int32)])
        x[2].append(np.eye(24)[features[3].astype(np.int32)])
        x[3].append(np.eye(10)[features[4].astype(np.int32)])
        x[4].append(np.ones(shape=(features[0].shape[0],1)))
    converted_data = np.array([np.array(f, dtype=object) for f in x], dtype=object)
    return converted_data

def data_conversion_test(df, tid_col):
    """Converts input panda dataframe to one-hot-encoded Numpy array (locations are still in float).
    
    Returns array of length 5 (lat-lon, day, hour, category,mask, tid, label)
    Each array has 2052 values, because there are 2052 trajectories
    """
    

    x = [[] for i in ['lat_lon', 'day', 'hour', 'category','mask','tid','label']]
    for tid in df[tid_col].unique():
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, ['lat', 'lon', 'day', 'hour', 'category','tid','label']].values)
        loc_list = []
        for i in range(0, len(traj)):
            lat = traj['lat'].values[i]
            lon = traj['lon'].values[i]
            loc_list.append(np.array([lat, lon], dtype=np.float64))
        x[0].append(loc_list)
        x[1].append(np.eye(7)[features[2].astype(np.int32)])
        x[2].append(np.eye(24)[features[3].astype(np.int32)])
        x[3].append(np.eye(10)[features[4].astype(np.int32)])
        x[4].append(np.ones(shape=(features[0].shape[0],1)))
        x[5].append(features[5][0].astype(np.int32))
        x[6].append(len(loc_list))
    converted_data = np.array([np.array(f, dtype=object) for f in x], dtype=object)
    return converted_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="/Users/jh/github/LSTM-TrajGAN/data/dev_train_encoded_final.csv")
    parser.add_argument("--save_path", type=str, default="/Users/jh/github/LSTM-TrajGAN/data/final_train.npy")
    parser.add_argument("--tid_col", type=str, default="tid")
    args = parser.parse_args()
    
    df = pd.read_csv(args.load_path)
    converted_data = data_conversion(df, args.tid_col)
    np.save(args.save_path, converted_data)