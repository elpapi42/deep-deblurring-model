#!/usr/bin/python
# coding=utf-8

"""
Run all the available download scripts.

"""

import os

from deblurrer.scripts.datasets import generate_csv, generate_tfrecord
from deblurrer.scripts.datasets import kaggle_blur


def run(path):
    """
    Run all the datasets download scripts.

    Generates .csv and .tfrecords

    Args:
        path (str): Path conting datasets folders.

    """
    # Starts the downloads
    kaggle_blur.run(path)

    # Generate csv
    generate_csv.run(path)

    # Generate TFRecords
    generate_tfrecord.run(path)


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
