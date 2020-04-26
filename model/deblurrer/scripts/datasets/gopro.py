#!/usr/bin/python
# coding=utf-8

"""
Downloads the GoPro Dataset dataset training data.

The data must be downloaded to "/datasets/gopro"

The module must define the data extraction logic.

# You can run this on google colab for get faster downloads speeds

"""

import os
import zipfile
import requests


from deblurrer.scripts.datasets.download_gdrive import download


def extract(file_path, extract_path):
    """
    Extract if exists.

    Args:
        file_path (str): Path of the file to be extracted
        extract_path (str): Path to copy the extracted files

    Returns:
        True if extracted successfully, False otherwise

    """
    if (os.path.exists(file_path) and os.path.isfile(file_path)):
        with zipfile.ZipFile(file_path, 'r') as compressed:
            compressed.extractall(extract_path)
            compressed.close()

            return True

    return False




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