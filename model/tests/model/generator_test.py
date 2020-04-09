#!/usr/bin/python
# coding=utf-8

"""Test suit for Generator sub-package."""

import tensorflow as tf

from deblurrer.model.generator import MobileNetV2Backbone, ConvBlock, FPNConvBlock, FPN, FPNGenerator


def test_mobilenetv2_backbone():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance backbone with default args
    backbone = MobileNetV2Backbone()

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    outputs = backbone(inputs)

    # asserts number of outputs
    assert len(outputs) == 5

    # Check shape of first, middel and last
    assert outputs[0].shape == (1, 64, 64, 128)
    assert outputs[2].shape == (1, 16, 16, 128)
    assert outputs[4].shape == (1, 8, 8, 128)


def test_convblock():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance block
    convblock = ConvBlock(3, 1, True)

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    outputs = convblock(inputs)

    # Check shape
    assert outputs.shape == (1, 256, 256, 3)


def test_fpn_convblock():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance block
    block = FPNConvBlock(3, 1)

    # Fake random image tensor
    input_a = tf.random.uniform([1, 256, 256, 3], seed=1)
    input_b = tf.random.uniform([1, 128, 128, 3], seed=1)

    outputs = block([input_b, input_a])

    # Check shape
    assert outputs.shape == (1, 256, 256, 3)


def test_fpn_model():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance block with default channels
    model = FPN()

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    outputs = model(inputs)

    # asserts number of outputs
    assert len(outputs) == 5

    # Check shape of first, middel and last
    assert outputs[0].shape == (1, 8, 8, 128)
    assert outputs[2].shape == (1, 16, 16, 128)
    assert outputs[4].shape == (1, 64, 64, 128)


def test_fpn_generator():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance block
    model = FPNGenerator()

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    outputs = model(inputs)

    # Check shape
    assert outputs.shape == (1, 256, 256, 3)