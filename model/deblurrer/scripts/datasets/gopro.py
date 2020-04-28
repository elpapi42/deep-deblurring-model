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
import shutil

from deblurrer.scripts.datasets.download_gdrive import run as gdrive_download


def refactor_folder(path):
    """
    Refactor dataset folder for be structered as sharp/blurred images.

    Args:
        path (str): The path where the function will operate

    """
    blur_path = path/'blur'
    sharp_path = path/'sharp'
    ops_path = path/'GOPRO'/'GOPRO_3840FPS_AVG_3-21'

    # Create folders
    blur_path.mkdir(parents=True, exist_ok=True)
    sharp_path.mkdir(parents=True, exist_ok=True)

    # Scan and copy imagesfrom the folders to the sharp/blur folders
    for data_slice in [ops_path/'train', ops_path/'test']:
        data_blur = data_slice/'blur'
        data_sharp = data_slice/'sharp'

        for dir_path, dest_path in [[data_blur, blur_path], [data_sharp, sharp_path]]:
            for img_stack in [dir for dir in dir_path.iterdir() if dir.is_dir()]:
                for image in [img for img in img_stack.iterdir() if img.suffix is '.png']:
                    dest = dest_path/'{folder}_{file}'.format(
                        folder=image.parent.stem,
                        file='{name}.png'.format(name=image.stem),
                    )
                    shutil.copy(image, dest)


def run(credentials_path, download_path):
    """
    Downloads gopro dataset from gdrive.

    Args:
        credentials_path (str): path from where to retrieve/save the gdriv credentials
        download_path (str): Where to create the download folder for this dataset
    """
    gdrive_download(
        gdrive_id='1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W',
        dataset_name='gopro',
        credentials_path=credentials_path,
        download_path=download_path,
    )

    refactor_folder(download_path/'gopro')


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