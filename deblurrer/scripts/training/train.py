#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.scripts.training import Tester, Trainer
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import DoubleScaleDiscriminator


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

    #with strategy.scope():

    # Setup float16 mixed precision
    if (int(os.environ.get('USE_MIXED_PRECISION'))):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    trainer = Trainer(
        FPNGenerator(int(os.environ.get('FPN_CHANNELS'))),
        DoubleScaleDiscriminator(),
        tf.keras.optimizers.Adam(float(os.environ.get('GEN_LR'))),
        tf.keras.optimizers.Adam(float(os.environ.get('DISC_LR'))),
    )

    trainer.train(valid_dataset, 4, valid_dataset=valid_dataset, verbose=True)


if (__name__ == '__main__'):
    # Get the path to the tfrcords folder
    path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__),
                    ),
                ),
            ),
        ),
        os.path.join('datasets', 'tfrecords'),
    )

    run(path)
