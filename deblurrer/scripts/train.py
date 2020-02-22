#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.

"""

import os
import time

from deblurrer.scripts.datasets.generate_dataset import get_dataset


def run(path):
    """
    Run the training script.

    Args:
        path (str): path from where to load tfrecords
    """
    # Create train dataset
    train_dataset = get_dataset(
        path,
        name='train',
        batch_size=int(os.environ.get('BATCH_SIZE')),
    )

    # Create validation dataset
    valid_dataset = get_dataset(
        path,
        name='valid',
        batch_size=int(os.environ.get('BATCH_SIZE')),
    )

    # Instantiate model and run training
    # Mock training
    for example in train_dataset.take(10):
        print('training')
        time.sleep(0.001)


if (__name__ == '__main__'):
    # Get the path to the tfrcords folder
    path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
        os.path.join('datasets', 'tfrecords'),
    )

    run(path)
