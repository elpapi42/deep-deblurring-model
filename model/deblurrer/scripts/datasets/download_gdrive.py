#!/usr/bin/python
# coding=utf-8

"""Download script for arbitrary gdrive files."""

import io
import pickle
import os
import pathlib

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from tqdm import tqdm

def download(gdrive_id, file, credentials_path, block_size=100):
    """
    Download the file from gdrive and writes it to file

    Args:
        gdrive_id (str): Id of the file in gdrive
        file (str): file path/name where to write the download contents
        credentials_path (str): path from where to retrieve/save the gdriv credentials
        chunk_size (int): Size of download chunks in MB
    """
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    token_pickle = os.path.join(credentials_path, 'token.pickle')
    if os.path.exists(token_pickle):
        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    creds_json = os.path.join(credentials_path, 'credentials.json')
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_json, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_pickle, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    # Request file metadata
    request = service.files().get_media(fileId=gdrive_id)

    # File to write the downladed data
    fh = open(file, "wb")

    # Streamer and writer of the file
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*block_size)

    # progress bar
    bar = tqdm(
        desc=file.stem,
        total=100,
    )

    # Loop download chunks
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        bar.n = int(status.progress() * 100)
        bar.refresh()


def run(gdrive_id, dataset_name, credentials_path, download_path):
    """
    Downloads dataset from gdrive.

    Args:
        gdrive_id (str): Id of the file in gdrive
        dataset_name (str): arbitrary name for name folders and files
        credentials_path (str): path from where to retrieve/save the gdriv credentials
        download_path (str): Where to create the download folder for this dataset
    """
    # Logs
    print('Downloading {name}'.format(name=dataset_name))

    download_path = pathlib.Path(download_path)
    file_name = download_path/dataset_name/'{name}.rar'.format(name=dataset_name)

    if (not file_name.exists()):
        os.mkdir(download_path/dataset_name)

        download(
            gdrive_id=gdrive_id,
            file=file_name,
            credentials_path=credentials_path,
            block_size=1
        )

if (__name__ == '__main__'):
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
   
    run('1-tLJSsdRVbi1OJLfF77ZrzrLdD-sYOCp', 'gopro', creds, data)