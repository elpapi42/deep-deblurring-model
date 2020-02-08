#!/usr/bin/python
# coding=utf-8

"""
Downloads the kaggle blur dataset training data.

The data must be downloaded to "/datasets/kaggle_blur"

The module must define the data extraction logic.

# You can run this on google colab for get faster downloads speeds

"""

import os

from kaggle import api


def refactor_folder(path):
    """Refactor dataset folder for be structered as sharp/blurred images."""
    old_sharp_path = os.path.join(path, 'old_sharp')
    new_sharp_path = os.path.join(path, 'sharp')
    blur_path = os.path.join(path, 'blur')

    os.rename(
        new_sharp_path,
        old_sharp_path,
    )

    os.mkdir(new_sharp_path)
    os.mkdir(blur_path)

    images = os.listdir(old_sharp_path)
    for image in images:
        os.rename(
            os.path.join(old_sharp_path, image),
            os.path.join(new_sharp_path, image.split('_')[0]),
        )



if (__name__ == '__main__'):
    folder_path = os.path.join(
        os.path.join(
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
        ),
        'kaggle_blur',
    )

    api.dataset_download_cli(
        'kwentar/blur-dataset',
        path=folder_path,
        unzip=True,
    )

    refactor_folder(folder_path)
