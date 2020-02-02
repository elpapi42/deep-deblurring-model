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
    source_url = 'https://storage.googleapis.com/kaggle-data-sets/270005/579020/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1580744780&Signature=Kr6SiZWo9gr1gPKiEjfi57zJUxhpDMS0TMLF27eSRcmMIeG3imF%2FjRtTCunZw4isKWbB%2BsWnQIeEN6nNKz6unLxkYbxCnu5z8kc%2FxVNNcganxaDLEEWHN%2F0PYt1M4kJp37up2gZuuF8%2BtpGKQ%2B0To8o1HlQ4qoJjv73bSy1lrrz1GFtgAyje4WsMq0l0wdoBYnsXLRDRPvewy7%2FZpYd9rsyrSUq3HH2OkZoyeisWKalxxEVG0AwbN6Ue31GKnZKSfrQs28IcJST9povRxUJLc7V9zZ6pPzJ5%2F3he8b2ZW31sBzQ6KrQtt9X6vx2oSSrYg7FWyT4nfIEMpDuuv20Ulw%3D%3D&response-content-disposition=attachment%3B+filename%3Dblur-dataset.zip'
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

