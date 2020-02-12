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

import pandas as pd


def run():
    """
    Generates .csv with sharp/blur image pairs.

    Args:
        path (str): Path to the sharp/blur folders

    """
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

    csv_path = os.path.join(folder_path, 'dataset.csv')

    if (not (os.path.exists(csv_path) and os.path.isfile(csv_path))):
        # list possible datasets subfolders
        ds_folders = os.listdir(folder_path)

        dataset = pd.DataFrame()

        for dset in ds_folders:
            dset_path = os.path.join(folder_path, dset)

            sharp_path = os.path.join(dset_path, 'sharp')
            blur_path = os.path.join(dset_path, 'blur')

            # Get names of kaggle blur images
            sharp_list = os.listdir(sharp_path)
            blur_list = os.listdir(blur_path)

            # Builds dataframe with kaggle blur image pairs paths
            dataframe = pd.DataFrame()
            dataframe['sharp'] = sharp_list
            dataframe['sharp'] = sharp_path + dataframe['sharp']
            dataframe['blur'] = blur_list
            dataframe['blur'] = blur_path + dataframe['blur']

            # loads, updates and writes the new gen dataframe to the full dataset csv
            if (not dataset.empty):
                dataset.append(dataframe)
            else:
                dataset = dataframe

        dataset.to_csv(csv_path, index=None)






















    """
    

    

   

    # loads, updates and writes the new gen dataframe to the full dataset csv
    
        dataset = pd.read_csv(csv_path)
        dataset.append(dataframe)
        dataset.to_csv(csv_path, index=None)
    else:
        dataframe.to_csv(csv_path, index=None)
    """

if (__name__ == '__main__'):
    run()
