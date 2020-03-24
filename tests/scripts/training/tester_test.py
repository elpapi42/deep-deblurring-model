#!/usr/bin/python
# coding=utf-8

"""Test suit for Tester class."""

import pytest
import tensorflow as tf

from deblurrer.scripts.training import Tester
from deblurrer.model import DoubleScaleDiscriminator, FPNGenerator


@pytest.fixture()
def dataset():
    """
    Mock images dataset.

    Returns:
        tensorflow dataset
    """
    def stack(images):
        return {
            'sharp': images,
            'blur': images,
        }

    images = tf.random.normal([10, 32, 32, 3], seed=1)

    dset = tf.data.Dataset.from_tensor_slices(images)
    dset = dset.map(stack)
    dset = dset.batch(2)

    return dset

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

    assert gen_loss == 0.45626307


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

    assert gen_loss == 0.405701