#!/usr/bin/python
# coding=utf-8

"""
Downloads the training data.

The data must be stored in "/dataset"

The modules must define the data extraction logic.

If the full training data is a composition of different datasets,
this module must be refactored into a python package,
where each package module,
defines the download and extraction logic of each data split.

# You can run this on google colab for get faster downloads speeds

"""

import os
import zipfile
import requests

from tqdm import tqdm


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


def download(source_url, download_path):
    """
    Download the file at source_url and stores it at download_path.

    Args:
        source_url (str): URL from where pull the file
        download_path (str): Local path for store the downloaded file

    Returns:
        True if file was downloaded, False otherwise

    """
    if (not (os.path.exists(download_path) and os.path.isfile(download_path))):
        resp = requests.get(source_url, stream=True)

        total_size = int(resp.headers.get('content-length', 0))
        block_size = 16384
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(download_path, 'wb') as stream_file:
            for block in resp.iter_content(block_size):
                progress_bar.update(len(block))
                stream_file.write(block)

            progress_bar.close()
            stream_file.close()

            if (total_size != 0 and progress_bar.n != total_size):
                print(total_size, progress_bar.n)
                return False

            return True

    return False


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


def preprocess():
    """Restructure the downloaded data into two folders: sharp and blur."""
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

    print(folder_path)

    # Create Dataset folder if not exists
    #if (not os.path.exists(folder_path)):
    #    os.mkdir(folder_path)


def execute():
    folder_path = create_folder()
    
    # Download link and download path
    source_url = 'https://storage.googleapis.com/kaggle-data-sets/270005/579020/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1581175497&Signature=d8zlv25W9ZY4%2BOuHOK3JA9w8XBK8GDnViFH6IKbiaPawON%2F1m0UEz5RR7VXxNwX0vl17SAiA9pggpCcGwQpWi%2BZVnGuQV721UkNV3g7LmaSup805uucL1JNEA1NeE4tG0YfodlUK0cz0jU2q21QDMavA02WJWln0mjKgjvCBgdkvvJ1tuLK8GoQ6LzeZQ0tf20ZTy6e%2BmHR%2F2ywU09bD%2Fd%2BLqGv5xzpZt2By2evtjFTBPoZfg1%2FSV6RNJCquu%2FpNBE9JFQaQSEMXUd3LOZ5essZp9JNK7QqKCnfhC5b30fbPlP9pdpVm89QVHLehwFmOM1sgdEa%2FsMtxc6JXwbz4FA%3D%3D&response-content-disposition=attachment%3B+filename%3Dblur-dataset.zip'
    download_path = os.path.join(folder_path, 'blur.zip')

    # download blur-dataset
    downloaded = download(source_url, download_path)
    if (not downloaded):
        print('Error on download or file already downloaded')

    if (downloaded):
        # Extract blur-dataset
        print('Extracting files')
        if (not extract(download_path, folder_path)):
            print('Download path does not exists')
        print('Extraction succesful')
    
    preprocess()


if (__name__ == '__main__'):
    execute()

