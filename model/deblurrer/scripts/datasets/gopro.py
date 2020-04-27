#!/usr/bin/python
# coding=utf-8

"""
Downloads the GoPro Dataset dataset training data.

The data must be downloaded to "/datasets/gopro"

The module must define the data extraction logic.

# You can run this on google colab for get faster downloads speeds

"""

import os
import pathlib

from deblurrer.scripts.datasets.download_gdrive import run as gdrive_download


def refactor_folder(path):
    """
    Refactor dataset folder for be structered as sharp/blurred images.

    Args:
        path (str): The path where the function will operate

    """
    pass


def run(credentials_path, download_path):
    """
    Downloads gopro dataset from gdrive.

    Args:
        credentials_path (str): path from where to retrieve/save the gdriv credentials
        download_path (str): Where to create the download folder for this dataset
    """
    gdrive_download.run(
        gdrive_id='1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W',
        dataset_name='gopro',
        credentials_path=credentials_path,
        download_path=download_path,
    )

    refactor_folder(download_path/dataset_name)


if (__name__ == '__main__'):
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