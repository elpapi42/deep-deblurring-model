#!/usr/bin/python
# coding=utf-8

"""Test suit for Generator sub-package."""

import tensorflow as tf

from deblurrer.model.generator import MobileNetV2Backbone


def test_mobilenetv2_backbone():
    # Instance backbone with default args
    backbone = MobileNetV2Backbone()

    # Seed random ops of tf
    tf.random.set_seed(1)

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    outputs = backbone(inputs)

    # asserts number of outputs
    assert len(outputs) == 5

    # Check shape of first, middel and last
    assert outputs[0].shape == (1, 64, 64, 128)
    assert outputs[2].shape == (1, 16, 16, 128)
    assert outputs[4].shape == (1, 8, 8, 128)
