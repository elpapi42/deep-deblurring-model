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
            'dataset',
        ),
        'kaggle_blur',
    )

    api.dataset_download_cli(
        'kwentar/blur-dataset',
        path=folder_path,
        unzip=True,
    )
