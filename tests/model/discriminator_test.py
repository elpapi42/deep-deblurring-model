#!/usr/bin/python
# coding=utf-8

"""Test suit for Discriminator related models."""

import tensorflow as tf

from deblurrer.model.discriminator import LocalDiscriminator, GlobalDiscriminator, LeakyConvBlock, DoubleScaleDiscriminator


def test_local_discriminator():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance model with default args
    discrim = LocalDiscriminator()

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    # Feeds faked sharp/blur pair
    outputs = discrim(
        {
            'sharp': inputs,
            'generated': inputs,
        }
    )

    # Check shape
    assert outputs.shape == (1, 1)


def test_leaky_convblock():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance block
    block = LeakyConvBlock(6, 1, 1, 0.2)

    # Fake random image tensor
    inputs = tf.random.uniform([1, 256, 256, 3], seed=1)

    # Feeds faked sharp/blur pair
    outputs = block(inputs)

    # Check shape
    assert outputs.shape == (1, 256, 256, 6)


def test_global_discriminator():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance model with default args
    discrim = GlobalDiscriminator()

    # Fake random image tensor
    inputs = tf.random.uniform([2, 256, 256, 3], seed=1)

    # Feeds faked sharp/blur pair
    outputs = discrim(
        {
            'sharp': inputs,
            'generated': inputs,
        }
    )

    # Check shape
    assert outputs.shape == (2, 1)


def test_double_scale_discriminator():
    # Seed random ops of tf
    tf.random.set_seed(1)

    # Instance model with default args
    discrim = DoubleScaleDiscriminator()

    # Fake random image tensor
    inputs = tf.random.uniform([2, 256, 256, 3], seed=1)

    # Feeds faked sharp/blur pair
    outputs = discrim(
        {
            'sharp': inputs,
            'generated': inputs,
        }
    )

    # Check shape
    assert outputs.shape == (2, 1)
