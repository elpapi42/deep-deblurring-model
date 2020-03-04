#!/usr/bin/python
# coding=utf-8

"""Test suit for custom losses module."""

from deblurrer.model.losses import ragan_ls_loss, generator_loss


def test_ragan_ls_loss():
    assert ragan_ls_loss is not None


def test_generator_loss():
    assert generator_loss is not None
