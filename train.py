import sys
import pandas as pd
import numpy as np

from model import LSTM_TrajGAN

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    n_batch_size = int(sys.argv[2])
    n_sample_interval = int(sys.argv[3])
    
    latent_dim = 100
    #max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon":2,"day":7,"hour":24,"category":10,"mask":1}
    
    tr = pd.read_csv('data/train_latlon_tapas.csv')
    te = pd.read_csv('data/test_latlon_tapas.csv')
    
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    scale_factor=max(max(abs(tr['lat'].max() - lat_centroid),
                         abs(te['lat'].max() - lat_centroid),
                         abs(tr['lat'].min() - lat_centroid),
                         abs(te['lat'].min() - lat_centroid),
                        ),
                     max(abs(tr['lon'].max() - lon_centroid),
                         abs(te['lon'].max() - lon_centroid),
                         abs(tr['lon'].min() - lon_centroid),
                         abs(te['lon'].min() - lon_centroid),
                        ))
    
    x_test = np.load('data/final_test_original.npy',allow_pickle=True)
    x_train = np.load('data/final_train_tapas.npy',allow_pickle=True)
    
    max_length=0
    for x in range(0, len(x_test[0])):
        if max_length < len(x_test[0][x]):
            max_length = len(x_test[0][x])
        else:
            pass
    for x in range(0, len(x_train[0])):
        if max_length < len(x_train[0][x]):
            max_length = len(x_train[0][x])
        else:
            pass

    print('Max length: ', max_length)

    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor)
    
    gan.train(epochs=n_epochs, batch_size=n_batch_size, sample_interval=n_sample_interval)