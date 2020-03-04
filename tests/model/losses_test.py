#!/usr/bin/python
# coding=utf-8

"""Test suit for custom losses module."""

from deblurrer.model.losses import ragan_ls_loss, generator_loss


def test_ragan_ls_loss():
    real_pred = [[0.75, 0.5, 0.95]]
    fake_pred = [[0.15, 0.45, 0.25]]

    loss, real, fake = ragan_ls_loss(real_pred, fake_pred)

    assert loss.shape == []
    assert loss == 0.69338644


def test_generator_loss():
    fake_pred = [[0.95, 0.6, 0.75]]

    loss = generator_loss(fake_pred)

    assert loss.shape == []
    assert loss == 0.2832668
