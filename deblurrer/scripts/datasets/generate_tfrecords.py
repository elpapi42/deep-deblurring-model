#!/usr/bin/python
# coding=utf-8

"""
Use the .csv for load, preprocess and record the images to a tfrecord file.

the csv must bein datasets/ folder
the tfrecords will be saved to the same location

"""

import os

import tensorflow as tf
import pandas as pd


def image_example(sharp, blur):
    """Create a dictionary with sharp and blur images."""
    feature = {
        'sharp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sharp])),
        'blur': tf.train.Feature(bytes_list=tf.train.BytesList(value=[blur])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_tfrecord_from_dataframe(path, df):
    """
    Generate tfrecord from the contents of dataframe.

    Args:
        path (str): filepath to store .tfrecord file
        df (pandas.Dataframe): dataframe with sharp/blur image pairs
    """
    with tf.io.TFRecordWriter(path) as writer:
        for _, columns in df.iterrows():
            # Open sharp image file
            with open(columns['sharp'], 'rb') as sharp_file:
                sharp = sharp_file.read()

            # Open blur image file
            with open(columns['blur'], 'rb') as blur_file:
                blur = blur_file.read()

            # Serializes and write to the file
            tf_example = image_example(sharp, blur)
            writer.write(tf_example.SerializeToString())


def split_dataframe(df, splits):
    """
    Split dataframe into 'splits' equal parts.

    is possible than the last split is not uniform.

    Args:
        df (pandas.Dataframe): Dataframe to split
        splits (int): Number of splits

    Returns: List of dataframe splits
    """
    length = int(len(df.index) / splits)
    chunks = []

    for idx in range(splits):
        if (idx < splits - 1):
            chunks.append(
                df.iloc[idx * length:(idx + 1) * length, :],
            )
        else:
            chunks.append(
                df.iloc[idx * length:, :],
            )

    return chunks


def generate_tfrecord_from_csv(path, csv, splits=1):
    """
    Generate a tfrecord from a csv with sharp/blur pairs.

    Args:
        path (str): Where to write the resulting tfrecords splits
        csv (str): csv file path
        splits (int): Number of parts to split the dataset, default=1

    """
    # Get the name of the csv file, this will be used for name the tfrecords
    csv_name = os.path.split(csv)[1].split('.')[0]

    # Load and splits dataset
    df_splits = split_dataframe(pd.read_csv(csv), splits)

    # Writes to disk every csv split as tfrecords files
    for index, split in enumerate(df_splits):
        # Sets a path for store the current split
        store_path = os.path.join(
            path,
            '{name}_{id}.tfrecords'.format(name=csv_name, id=index),
        )

        # If the file does not exist, create it.
        if (not (os.path.exists(store_path) and os.path.isfile(store_path))):
            generate_tfrecord_from_dataframe(store_path, split)


def run(path):
    """
    Run the script.

    Args:
        path (str): Folder containing tfrecords and csv folders

    """
    # Folder to store tfrecords splits
    tfrecs_path = os.path.join(path, 'tfrecords')

    # Only executes if the tfrecords folder is not there
    if (not (os.path.exists(tfrecs_path) and os.path.isdir(tfrecs_path))):
        # Logs
        print('Generating TFRecords')

        # Make csv folder
        os.mkdir(tfrecs_path)

        # Csv files and its desired split count
        csv_files = [['train.csv', 4], ['valid.csv', 1], ['test.csv', 1]]

        for csv_name, splits in csv_files:
            # Logs
            print('Generating {name} TFRecords'.format(name=csv_name))

            # Builds the path of the next csv
            csv_file = os.path.join(os.path.join(path, 'csv'), csv_name)

            # Generate the tfrecord
            generate_tfrecord_from_csv(tfrecs_path, csv_file, splits)

        # Logs
        print('TFRecords succesfully generated')
    else:
        # Logs
        print('TFRecords already generated')


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
