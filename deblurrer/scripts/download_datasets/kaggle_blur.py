#!/usr/bin/python
# coding=utf-8

"""
Downloads the kaggle blur dataset training data.

The data must be downloaded to "/dataset/kaggle_blur"

The module must define the data extraction logic.

# You can run this on google colab for get faster downloads speeds

"""

import os

from kaggle import api


def create_folder():
    """
    Generate download path if it does not exist.

    Returns:
        Generated folder path

    """
    folder_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
        'dataset',
    )

    # Create Dataset folder if not exists
    if (not os.path.exists(folder_path)):
        os.mkdir(folder_path)

    return folder_path


if (__name__ == '__main__'):
    folder_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
        'dataset',
    )
    api.dataset_download_cli('kwentar/blur-dataset', path=folder_path, unzip=True)
