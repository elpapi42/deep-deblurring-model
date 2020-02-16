#!/usr/bin/python
# coding=utf-8

"""
Reads the tf records and put them into a TFRecordDataset.

the tfrecords must be in datasets/ folder

"""

import os

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn


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


def get_dataset_from_tfrecord(path, name, batch_size=8):
    """
    Return the fully transformed version of the dataset.

    Args:
        path (str): path to the folder containing the tfrecord to load
        name (str): name of the tfrecord to load w/o .tfrecord extension
        batch_size (int): size of the batch

    Returns:
        tf.data.Dataset with the full load and tranform pipeline
    
    """
    dataset = tf.data.TFRecordDataset(
        os.path.join(path, '{name}.tfrecords'.format(name=name)),
    )

    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Cache previous transformations into a file at the same dir than .tfrecord
    dataset = dataset.cache(
        os.path.join(path, '{name}.tfcache'.format(name=name)),
    )

    return dataset

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

    dataset = get_dataset_from_tfrecord(folder_path, 'test', batch_size=8)

    benchmark(dataset, num_epochs=1)

        
