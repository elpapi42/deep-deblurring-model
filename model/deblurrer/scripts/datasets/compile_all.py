#!/usr/bin/python
# coding=utf-8

"""
Run all the available download scripts.

"""

import os
import pathlib

from deblurrer.scripts.datasets import generate_csv, generate_tfrecord
from deblurrer.scripts.datasets import kaggle_blur, download_gdrive


def run(path):
    """
    Run all the datasets download scripts.

    Generates .csv and .tfrecords

    Args:
        path (str): Path conting datasets and credentials folders.

    """
    # Datasets and  credentials folders
    data = pathlib.Path(path)/'datasets'
    creds = pathlib.Path(path)/'credentials'

    # Starts the downloads
    kaggle_blur.run(data)
    download_gdrive.run(
        gdrive_id='1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W',
        dataset_name='gopro',
        credentials_path=creds,
        download_path=data,
    )

    # Generate csv
    generate_csv.run(data)

    # Generate TFRecords
    generate_tfrecord.run(data)


if (__name__ == '__main__'):
    # Get the path to the datasets folder
    folder_path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
    )

    run(folder_path)
