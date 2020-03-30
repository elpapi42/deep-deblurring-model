#!/usr/bin/python
# coding=utf-8

"""
Start the testing of the Model Architecture.

This module will eclusively contains testing logic.
"""

import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.scripts.training import Tester
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import DoubleScaleDiscriminator


def run(
    path,
    generator=None,
    discriminator=None,
    strategy=None,
    output_folder='',
):
    """
    Run the training script.

    Args:
        path (str): path from where to load tfrecords
        generator (tf.keras.Model): FPN Generator
        discriminator (tf.keras.Model): DS Discriminator
        strategy (tf.distributed.Strategy): Distribution strategy
        output_folder (str): Where to store images for performance test
    """
    # Create validation dataset
    test_dataset = get_dataset(
        path,
        name='test',
        batch_size=int(os.environ.get('BATCH_SIZE')),
    )

    # Setup float16 mixed precision
    if (int(os.environ.get('USE_MIXED_PRECISION'))):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    # If in colba tpu instance, use tpus, gpus otherwise
    colab_tpu = os.environ.get('COLAB_TPU_ADDR') is not None
    if (colab_tpu and (strategy is None)):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'),
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        # Convert dataset to distribute datasets
        test_dataset = strategy.experimental_distribute_dataset(test_dataset)
    elif (strategy is None):
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

    with strategy.scope():
        # Instantiate models and optimizers
        if (generator is None):
            generator = FPNGenerator(int(os.environ.get('FPN_CHANNELS')))
        if (discriminator is None):
            discriminator = DoubleScaleDiscriminator()

        tester = Tester(
            generator,
            discriminator,
            strategy,
            output_folder,
        )

        tester.test(
            test_dataset,
            verbose=True,
        )

    return generator, discriminator, strategy


if (__name__ == '__main__'):
    # Get the path to the tfrcords folder
    path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
    )

    tfrec_path = os.path.join(
        path,
        os.path.join('datasets', 'tfrecords'),
    )

    output_path = os.path.join(path, 'output')

    run(tfrec_path, output_folder=output_path)
