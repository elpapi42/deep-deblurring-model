#!/usr/bin/python
# coding=utf-8

"""Collection of fixtures used across the test suit."""

import pytest
import tensorflow as tf

@pytest.fixture()
def dataset():
    """
    Mock images dataset.

    Returns:
        tensorflow dataset
    """
    images = tf.random.normal([10, 2, 32, 32, 3], seed=1)

    dset = tf.data.Dataset.from_tensor_slices(images)
    dset = dset.batch(2)

    return dset


@pytest.fixture()
def loss_network():
    """
    Mock loss network.

    Returns:
        VGG19 based loss network
    """
    vgg19 = tf.keras.applications.VGG19(include_top=False)

    return tf.keras.Model(
        inputs=vgg19.inputs,
        outputs=vgg19.get_layer(name='block3_conv3').output,
    )
