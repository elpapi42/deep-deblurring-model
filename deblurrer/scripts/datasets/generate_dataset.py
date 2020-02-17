#!/usr/bin/python
# coding=utf-8

"""
Reads the tf records and put them into a TFRecordDataset.

the tfrecords must be in datasets/ folder

"""

import os
import re
import glob

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn

AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse(example):
    """
    Parse single example from tfrecord to actual image.

    this fn operates over a single sharp/blur pair
    if you know how to vectorize this, colaborate

    Args:
        example (tf.Tensor): sharp/blur tfrecord encoded strings

    Return: fully parsed, decoded and loaded sharp/blur pair(Rank 4)
    
    """
    feature_properties = {
        'sharp': tf.io.FixedLenFeature([], tf.string),
        'blur': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_example(example, feature_properties)

    # Decode sharp
    sharp = tf.image.decode_image(example['sharp'])
    sharp = tf.image.resize_with_pad(sharp, 1024, 1024)

    # Decode blur
    blur = tf.image.decode_image(example['blur'])
    blur = tf.image.resize_with_pad(blur, 1024, 1024)

    example = tf.stack([sharp, blur])

    return example


def transform(example):
    """
    Applies transforms to a batch of sharp/blur pairs.

    This fn is vectorized

    Args:
        example (tf.Tensor): ully parsed, decoded and loaded sharp/blur pair

    Returns: batch of transformed sharp/blur pairs tensor(Rank 5)
    
    """
    # Swaps batch dimension to be second, this make calcs easier in the future
    example = tf.transpose(example, [1, 0, 2, 3, 4])

    sharp, blur = tf.unstack(example)

    # Generates a random resolution
    rnd_size = tf.random.uniform([], minval=256, maxval=1440)

    # Resize images to a random res between 256 and 1440
    sharp = tf.image.resize(sharp, [rnd_size, rnd_size])
    blur = tf.image.resize(blur, [rnd_size, rnd_size])

    example = tf.stack([sharp, blur])

    # Scales to 0.0 - 1.0 range
    example = example / 256.0

    return example


def get_dataset_from_tfrecord(path, batch_size):
    """
    Return the fully transformed version of the dataset.

    Args:
        path (str): path to the folder containing the tfrecord to load
        batch_size (int): size of the batch

    Returns:
        tf.data.Dataset with the full load and tranform pipeline

    """
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(transform, num_parallel_calls=AUTOTUNE)

    # Cache previous transformations into a file at the same dir than .tfrecord
    dataset = dataset.cache(
        tf.strings.join(
            [tf.strings.split(path, '.')[0], tf.constant('.tfcache')],
        ),
    )

    return dataset


def get_interleave_dataset(path, name, batch_size=8):
    """
    Return an interleaved dataset.

    The dataset is composed of severals datasets
    named with suffix=name. Ex:

    train_01.tfrecords
    train_02.tfrecords

    with name='train'

    will load all the tfrecords suffixed with 'train' and interleave them

    Args:
        path (str): absolute path to the tfrecords folder
        name (str): suffix name to look for tf records

    Returns: interleaved dataset composed of all the matching tfrecords

    """
    # Find all the relevant tfrecord following the name suffix
    tfrecs = glob.glob(
        '{path}*.tfrecords'.format(path=os.path.join(path, name)),
    )

    batch_size = [batch_size for index in enumerate(tfrecs)]
    batch_size = tf.cast(batch_size, tf.int64)

    tfrecs = tf.data.Dataset.from_tensor_slices((tfrecs, batch_size))
    tfrecs = tfrecs.interleave(
        get_dataset_from_tfrecord,
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
    )

    return tfrecs


import time

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for _epoch_num in range(num_epochs):
        for _sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)


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

    dataset = get_interleave_dataset(os.path.join(folder_path, 'tfrecords'), 'test')
    #dataset = get_dataset_from_tfrecord(os.path.join(os.path.join(folder_path, 'tfrecords'), 'test.tfrecords'), batch_size=8)

    benchmark(dataset, 10)
