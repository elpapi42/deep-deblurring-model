#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os
import contextlib

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.scripts.training import Tester, Trainer
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import DoubleScaleDiscriminator


def run(
    path,
    generator=None,
    discriminator=None,
    gen_optimizer=None,
    disc_optimizer=None,
):
    """
    Run the training script.

    Args:
        path (str): path from where to load tfrecords
        generator (tf.keras.Model): FPN Generator
        discriminator (tf.keras.Model): DS Discriminator
        gen_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
        disc_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
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

    # Setup float16 mixed precision
    if (int(os.environ.get('USE_MIXED_PRECISION'))):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    # If in colba tpu instance, use tpus, gpus otherwise
    colab_tpu = os.environ.get('COLAB_TPU_ADDR') is not None
    if (colab_tpu):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'),
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        # Convert dataset to distribute datasets
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

    with strategy.scope():
        # Instantiate models and optimizers
        if (generator is None):
            generator = FPNGenerator(int(os.environ.get('FPN_CHANNELS')))
        if (discriminator is None):
            discriminator = DoubleScaleDiscriminator()
        if (gen_optimizer is None):
            gen_optimizer = tf.keras.optimizers.Adam(float(os.environ.get('GEN_LR')))
        if (disc_optimizer is None):
            disc_optimizer = tf.keras.optimizers.Adam(float(os.environ.get('DISC_LR')))

        trainer = Trainer(
            generator,
            discriminator,
            gen_optimizer,
            disc_optimizer,
            strategy,
        )

        trainer.train(
            train_dataset,
            int(os.environ.get('EPOCHS')),
            valid_dataset=valid_dataset,
            verbose=True,
        )

    return generator, discriminator, gen_optimizer, disc_optimizer


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
