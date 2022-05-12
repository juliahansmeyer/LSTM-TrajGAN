from re import X
from geojson import LineString, Feature, FeatureCollection, dump
import yaml
import pandas as pd
import numpy as np

if __name__ == '__main__':

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        geojson_path = dataset_configs['geojson_path']
        synthetic_path = dataset_configs['synthetic_csv_path']

    df = pd.read_csv(synthetic_path)

    coordinates = []
    type = []
    trip_ids = df[['tid']].values
    unique_trips = np.unique(trip_ids)

    for trip in unique_trips:
        df_temp = df[df['tid']==trip]
        lat= df_temp['lat']
        lon= df_temp['lon']
        coords = list(zip(lon,lat)) # order is lon-lat for visualisation in R
        coordinates.append(LineString(coords))
        type.append('synthetic')

    features = []

    for i in range(len(type)):
        features.append(Feature(geometry=coordinates[i], properties={"Type": type[i]}))

    feature_collection = FeatureCollection(features)
    with open(geojson_path, 'w') as f:
        dump(feature_collection, f)

