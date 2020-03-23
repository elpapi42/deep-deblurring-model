#!/usr/bin/python
# coding=utf-8

"""Test suit for Tester class."""

import pytest
import tensorflow as tf


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

    images = tf.random.normal([10, 32, 32, 3])

    dset = tf.data.Dataset.from_tensor_slices(images)
    dset = dset.map(stack)
    dset = dset.batch(2)

    return dset
