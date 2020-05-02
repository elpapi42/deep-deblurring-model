#!/usr/bin/python
# coding=utf-8

"""
Use all the available data for generate a .csv with blur/sharp images paths.

Every folder must contain two subfolder with the respective shar/blur images:

- <dataset_name>
-- sharp
--- img01.jpg
--- img02.jpg
-- blur
--- img01.jpg
--- img02.jpg

The image pairs must have the same name inside different folders
img01.jpg is present in blur and sharp folders

"""

import os
import pathlib

import pandas as pd


def run(path):
    """
    Generate train/valid/test.csv with sharp/blur image pairs.

    Args:
        path (str): Path containg datasets folders. csv will be stored here too

    """
    path = pathlib.Path(path)
    csv_path = path/'csv'

    if (not (csv_path.exists() and csv_path.is_dir())):
        # Creates csv folder
        csv_path.mkdir(parents=True, exist_ok=True)

        dataset = pd.DataFrame(columns=['sharp', 'blur'])

        for dset in path.iterdir():
            sharp_path = dset/'sharp'
            blur_path = dset/'blur'

            if (sharp_path.exists() and blur_path.exists()):
                # Builds dataframe with kaggle blur image pairs paths
                dataframe = pd.DataFrame()
                dataframe['sharp'] = [img for img in sharp_path.iterdir()]
                dataframe['blur'] = [img for img in blur_path.iterdir()]

                dataset = dataset.append(dataframe, ignore_index=True)

        # Split dataset
        train = dataset.sample(frac=0.8)
        dataset = dataset.drop(train.index)
        valid = dataset.sample(frac=0.5)
        test = dataset.drop(valid.index)

        # Writes to storage
        train.to_csv(csv_path/'train.csv', index=None)
        valid.to_csv(csv_path/'valid.csv', index=None)
        test.to_csv(csv_path/'test.csv', index=None)
        
        # Logs
        print('.csv files successfully generated')
    else:
        # Logs
        print('.csv files already generated')


if (__name__ == '__main__'):
    # Get the path to the datasets folder
    folder_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__),
                    ),
                ),
            ),
        ),
        'datasets',
    )

    run(folder_path)
