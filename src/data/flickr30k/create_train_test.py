"""
make sure to download the dataset for flickr30k before running this
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_test(dataset_path):
    df=pd.read_csv(dataset_path+'results.csv', sep='|')

    unique_images=df['image_name'].unique()

    train_images, test_images=train_test_split(unique_images, test_size=0.2, random_state=26)

    train_df=df[df['image_name'].isin(train_images)].copy()
    test_df=df[df['image_name'].isin(test_images)].copy()
    return train_df, test_df