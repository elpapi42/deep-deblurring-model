#!/usr/bin/python
# coding=utf-8

"""Test suit for Tester class."""

import tensorflow as tf

from deblurrer.scripts.training import Tester
from deblurrer.model import DoubleScaleDiscriminator, FPNGenerator
from tests.fixtures import dataset

def test_step_fn(dataset):
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    with strategy.scope():
        tester = Tester(
            FPNGenerator(16),
            DoubleScaleDiscriminator(),
            strategy,
        )

        for batch in dataset.take(1):
            gen_loss, disc_loss = tester.step_fn(batch)

    assert tf.cast(gen_loss, dtype=tf.float16) == 1.455
    assert tf.cast(disc_loss, dtype=tf.float16) == 5.0


def test_distributed_step(dataset):
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    with strategy.scope():
        tester = Tester(
            FPNGenerator(16),
            DoubleScaleDiscriminator(),
            strategy,
        )

        for batch in dataset.take(1):
            gen_loss, disc_loss = tester.distributed_step(batch)

    assert tf.cast(gen_loss, dtype=tf.float16) == 1.36793
    assert tf.cast(disc_loss, dtype=tf.float16) == 5.0054493


def test_gan_forward_pass(dataset):
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    with strategy.scope():
        tester = Tester(
            FPNGenerator(16),
            DoubleScaleDiscriminator(),
            strategy,
        )

        for batch in dataset.take(1):
            _, input_images = tf.unstack(batch, axis=1)
            real_out, fake_out, gen_images = tester.gan_forward_pass(batch)

    assert gen_images.shape == input_images.shape
    assert real_out['local'].shape == [2, 1]
    assert fake_out['global'].shape == [2, 1]
    assert tf.cast(fake_out['global'][0][0], dtype=tf.float16) == 0.4863


def test_get_loss_network(dataset):
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    with strategy.scope():
        tester = Tester(
            FPNGenerator(16),
            DoubleScaleDiscriminator(),
            strategy,
        )

        loss_network = tester.get_loss_network()

    assert len(loss_network.layers) == 10
    assert loss_network.get_layer(index=-1).name == 'block3_conv3'
