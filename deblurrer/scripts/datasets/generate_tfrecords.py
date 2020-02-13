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

def generate_tfrecord(path, csv_name):
    """
    Generates a tfrecord from a csv with sharp/blur paths.

    Args:
        path (str): From where to load the csv and store the tfrecord
        csv_name (str): name of the csv file

    """
    # Compose tfrecord storage path
    record_path = '{name}.tfrecords'.format(name=csv_name.split('.')[0])

    if (not (os.path.exists(record_path) and os.path.isfile(record_path))):
        # Load csv
        dataframe = pd.read_csv(os.path.join(path, csv_name))

        with tf.io.TFRecordWriter(os.path.join(path, record_path)) as writer:
            for _, columns in dataframe.iterrows():

                with open(columns['sharp'], 'rb') as sharp_file:
                    sharp = sharp_file.read()
                with open(columns['blur'], 'rb') as blur_file:
                    blur = blur_file.read()

                tf_example = image_example(sharp, blur)
                writer.write(tf_example.SerializeToString())


def run(folder_path):
    """
    Generates tfrecords with sharp/blur image pairs.

    Args:
        folder_path (str): Path conting .csv files.

    """
    data_splits = ['train.csv', 'valid.csv', 'test.csv']

    for split in data_splits:
        generate_tfrecord(folder_path, split)


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
