import pandas as pd
import argparse
import random
import numpy as np
from csv2npy import data_conversion
pd.options.mode.chained_assignment = None


def tapas_to_csv(df):


    #shape into TrajGAN format
    df = df.rename(columns={'X': 'lon', 'Y': 'lat', 'PERS_ID|integer':'label','TRIP_ID|integer':'tid'})

    #chose subset
    trip_ids = df[['tid']].values
    unique_trips = np.unique(trip_ids)
    subset_ids =  unique_trips[0:2000]
    df = df[df['tid'].isin(subset_ids)]

    #change features to same format as trajGAN approach
    len_df = len(df.index)
    day = random.choices(range(0, 7), k=len_df)
    hour = random.choices(range(0,24), k=len_df)
    category = random.choices(range(0,4), k=len_df)
    df['day'] = day
    df['hour'] = hour
    df['category'] = category
    df = df[['lat', 'lon', 'day', 'hour', 'category','label','tid']]

    #train test split
    trip_ids = df[['tid']].values
    unique_trips = np.unique(trip_ids)
    train_ids = unique_trips[int(len(unique_trips) * 0) : int(len(unique_trips) * 0.65)]
    test_ids = unique_trips[int(len(unique_trips) * 0.65) : int(len(unique_trips) * 1)]
    df_train = df[df['tid'].isin(train_ids)]
    df_test = df[df['tid'].isin(test_ids)]

    #save as csv
    df_train.to_csv('/Users/jh/github/LSTM-TrajGAN/data/train_latlon_tapas.csv', index=False)
    df_test.to_csv('/Users/jh/github/LSTM-TrajGAN/data/test_latlon_tapas.csv', index=False)

    #deviations for lat lon
    lat_centroid = (df_train['lat'].sum() + df_test['lat'].sum())/(len(df_train)+len(df_test))
    lon_centroid = (df_train['lon'].sum() + df_test['lon'].sum())/(len(df_train)+len(df_test))
    
    df_train['lat'] = df_train['lat'].sub(lat_centroid)
    df_train['lon'] = df_train['lon'].sub(lon_centroid)
    df_test['lat'] = df_test['lat'].sub(lat_centroid)
    df_test['lon'] = df_test['lon'].sub(lon_centroid)

    #save csv with deviation lat lon
    df_train.to_csv('/Users/jh/github/LSTM-TrajGAN/data/dev_train_encoded_final_tapas.csv', index=False)
    df_test.to_csv('/Users/jh/github/LSTM-TrajGAN/data/dev_test_encoded_final_tapas.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="/Users/jh/github/LSTM-TrajGAN/data/")
    args = parser.parse_args()
    
    tapas_df = pd.read_csv('/Users/jh/github/freemove/data/TAPAS/tapas_big_single_coordinates.csv')
    tapas_to_csv(tapas_df)

    df_train = pd.read_csv('/Users/jh/github/LSTM-TrajGAN/data/dev_train_encoded_final_tapas.csv')
    df_test = pd.read_csv('/Users/jh/github/LSTM-TrajGAN/data/dev_test_encoded_final_tapas.csv')
    converted_data_train = data_conversion(df_train, 'tid')
    converted_data_test = data_conversion(df_test, 'tid')

    np.save(args.save_path+'final_train_tapas.npy', converted_data_train)
    np.save(args.save_path+'final_test_tapas.npy', converted_data_test)
