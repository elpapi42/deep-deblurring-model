#!/usr/bin/python
# coding=utf-8

"""
Run all the available download scripts.

"""

import os
import pathlib

from deblurrer.scripts.datasets import generate_csv, generate_tfrecord
from deblurrer.scripts.datasets import kaggle_blur, gopro


def run(data, credentials):
    """
    Run all the datasets download scripts.

    Generates .csv and .tfrecords

    Args:
        data (str): Path conting datasets folders.
        credentials (str): Path for credentials folder

    """
    # Datasets and  credentials folders
    data = pathlib.Path(data)
    creds = pathlib.Path(credentials)

    # Starts the downloads
    kaggle_blur.run(data)
    gopro.run(
        credentials_path=creds,
        download_path=data,
    )

    # Generate csv
    generate_csv.run(data)

    # Generate TFRecords
    generate_tfrecord.run(data)


if (__name__ == '__main__'):
    # Get the path to the datasets folder
    path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
    )

    data = os.path.join(path, 'datasets')
    creds = os.path.join(path, 'credentials')

    run(data, creds)
