#!/usr/bin/python
# coding=utf-8

"""
Reads the tf records and put them into a TFRecordDataset.

the tfrecords must be in datasets/ folder

"""

import os
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


def get_dataset(path, name, batch_size=8):
    """
    Generate an interleaved dataset.

    The dataset is composed of several tfrecords
    named with suffix=name. Ex:

    train_01.tfrecords
    train_02.tfrecords

    with name='train'

    will load all the tfrecords suffixed with 'train' and interleave them

    Args:
        path (str): absolute path to the tfrecords folder
        name (str): suffix name to look for tf records
        batch_size (str): batch size of sub-datasets

    Returns:
        interleaved dataset composed of all the matching tfrecords

    """
    # Find all the relevant tfrecord following the name suffix
    tfrecs = glob.glob(
        '{path}*.tfrecords'.format(path=os.path.join(path, name)),
    )

    print(path)

    # Creates a dataset listing out tfrecord files
    dataset = tf.data.Dataset.from_tensor_slices(tfrecs)

    # Interleave the tfrecord files contents into a single fast dataset
    dataset = dataset.interleave(
        lambda tfrec: tf.data.TFRecordDataset(tfrec),
        cycle_length=len(tfrecs),
        block_length=batch_size,
        num_parallel_calls=AUTOTUNE,
    )

    # Parse, batch and transform
    dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(transform, num_parallel_calls=AUTOTUNE)

    # Cache transforms
    dataset = dataset.cache(
        os.path.join(path, '{name}_cache'.format(name=name))
    )

    # Prefetch data
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


import time

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for _ in range(num_epochs):
        for _ in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)

def timeit(ds, steps=1000, batch_size=8):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(batch_size*steps/duration))


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

    dataset = get_dataset(os.path.join(folder_path, 'tfrecords'), 'train', batch_size=8)
    #dataset = get_dataset_from_tfrecord(os.path.join(os.path.join(folder_path, 'tfrecords'), 'train_0.tfrecords'), batch_size=16)

    #benchmark(dataset, 2)

    dataset = dataset.repeat()

    timeit(dataset, 100, batch_size=8)
