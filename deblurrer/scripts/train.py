#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os

import tensorflow as tf
import numpy as np

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import LocalDiscriminator, GlobalDiscriminator


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

    # If the machine executing the code has TPUs, use them
    if (True):
        strategy = tf.distribute.MirroredStrategy()
    else:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'),
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    # Instantiates the model for training
    with strategy.scope():
        model = GlobalDiscriminator()

    #print(model.fpn.backbone.backbone.summary())
    # Instantiate model and run training
    # Mock training
    for example in train_dataset.take(1):
        print(np.shape(example['blur']))
        print(model(example))

    

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
