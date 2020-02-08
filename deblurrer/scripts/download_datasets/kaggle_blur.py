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
    old_defocus_path = os.path.join(path, 'defocused_blurred')
    old_motion_path = os.path.join(path, 'motion_blurred')
    new_sharp_path = os.path.join(path, 'sharp')
    new_blur_path = os.path.join(path, 'blur')

    os.rename(
        new_sharp_path,
        old_sharp_path,
    )

    os.mkdir(new_sharp_path)
    os.mkdir(new_blur_path)

    images = os.listdir(old_sharp_path)
    for sharp_image in images:
        os.rename(
            os.path.join(old_sharp_path, sharp_image),
            os.path.join(new_sharp_path, sharp_image.split('_')[0]),
        )

    images = os.listdir(old_defocus_path)
    for defocus_image in images:
        os.rename(
            os.path.join(old_defocus_path, defocus_image),
            os.path.join(new_blur_path, defocus_image.split('_')[0]),
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
