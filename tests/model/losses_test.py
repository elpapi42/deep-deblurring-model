#!/usr/bin/python
# coding=utf-8

"""Test suit for custom losses module."""

import tensorflow as tf
import pytest

from deblurrer.model.losses import ragan_ls_loss
from deblurrer.model.losses import discriminator_loss
from deblurrer.model.losses import generator_loss
from deblurrer.model.losses import feature_reconstruction_loss
from tests.fixtures import loss_network


def test_ragan_ls_loss():
    pred = tf.constant([[0.5], [0.55]])

    loss = ragan_ls_loss(pred, real_preds=True)

    #assert loss == 0.22625
    assert loss.shape == []


def test_discriminator_loss():
    pred = {
        'local': tf.constant([[0.75, 0.5, 0.95]]),
        'global': tf.constant([[0.75, 0.5, 0.95]]),
    }

    loss = discriminator_loss(pred, True)

    assert loss.shape == []
    #assert loss == 4.9108334


def test_generator_loss(loss_network):
    fake_pred = {
        'local': tf.constant([[0.15, 0.45, 0.25]]),
        'global': tf.constant([[0.75, 0.5, 0.95]]),
    }

    # Fake image, will be generated and sharp image
    gen_input = tf.random.uniform([4, 32, 32, 3], seed=1)
    sharp_input = tf.random.uniform([4, 32, 32, 3], seed=2)

    loss = generator_loss(gen_input, sharp_input, fake_pred, loss_network)

    assert loss.shape == []
    #assert tf.cast(loss, dtype=tf.float16) == 0.08987017


def test_feature_reconstruction_loss(loss_network):
    # Fake image, will be generated and sharp image
    gen_input = tf.random.uniform([4, 32, 32, 3], seed=1)
    sharp_input = tf.random.uniform([4, 32, 32, 3], seed=2)

    loss = feature_reconstruction_loss(gen_input, sharp_input, loss_network)

    assert loss.shape == []
    assert tf.cast(loss, dtype=tf.float16) == 0.0011661573
