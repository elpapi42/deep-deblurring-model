#!/usr/bin/python
# coding=utf-8

"""Test suit for Trainer class."""

import tensorflow as tf
import numpy as np

from deblurrer.scripts.training import Trainer
from deblurrer.model import DoubleScaleDiscriminator, FPNGenerator
from tests.fixtures import dataset

def test_trainer_step_fn(dataset):
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    trainer = Trainer(
        FPNGenerator(16),
        DoubleScaleDiscriminator(),
        tf.keras.optimizers.Adam(0.1),
        tf.keras.optimizers.Adam(0.1),
        strategy,
    )

    pre_weights = trainer.generator.trainable_variables[0].numpy()

    for batch in dataset:
        gen_loss, disc_loss = trainer.step_fn(batch, training=True)

    post_weights = trainer.generator.trainable_variables[0].numpy()

    # Check if weights receive updates
    weights_delta = np.abs(pre_weights - post_weights)
    assert np.mean(weights_delta) > 0.0


def test_trainer_train(dataset):
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    with strategy.scope():
        trainer = Trainer(
            FPNGenerator(16),
            DoubleScaleDiscriminator(),
            tf.keras.optimizers.Adam(0.1),
            tf.keras.optimizers.Adam(0.1),
            strategy,
        )

        pre_weights = trainer.generator.trainable_variables[0].numpy()

        trainer.train(dataset, 1)

        post_weights = trainer.generator.trainable_variables[0].numpy()

    # Check if weights receive updates
    weights_delta = np.abs(pre_weights - post_weights)
    assert np.mean(weights_delta) > 0.0
