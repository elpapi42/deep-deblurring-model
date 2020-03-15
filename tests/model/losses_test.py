#!/usr/bin/python
# coding=utf-8

"""Test suit for custom losses module."""

import tensorflow as tf

from deblurrer.model.losses import ragan_ls_loss
from deblurrer.model.losses import discriminator_loss
from deblurrer.model.losses import generator_loss


def test_ragan_ls_loss():
    pred = tf.constant([[0.5], [0.55]])

    loss = ragan_ls_loss(pred, real_preds=True)

    assert loss == 0.22625
    assert loss.shape == []


def test_discriminator_loss():
    real_pred = tf.constant([[0.75, 0.5, 0.95]])
    fake_pred = tf.constant([[0.15, 0.45, 0.25]])

    loss = discriminator_loss(real_pred, fake_pred)

    assert loss.shape == []
    assert loss == 1.7675


def test_generator_loss():
    fake_pred = [[0.95, 0.6, 0.75]]

    loss = generator_loss(fake_pred)

    assert loss.shape == []
    assert loss == 0.2832668
