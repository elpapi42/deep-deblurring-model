#!/usr/bin/python
# coding=utf-8

"""Defines the custom loss functions used at train time."""

import tensorflow as tf


def ragan_ls_loss(real_pred, fake_pred):
    """
    Compute the RaGAN-LS loss of DScaleDiscrim.

    It is a two-term binary crossentropy loss
    each one associated with one of the DoubleScaleDiscriminator heads
    Author call it RaGAN-LS loss

    Args:
        real_pred (tf.Tensor): discriminator over real images
        fake_pred (tf.Tensor): discrim over generated images

    Returns:
        Total loss over real and fake images
    """
    bin_cross = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    real_loss = bin_cross(tf.ones_like(real_pred), real_pred)
    fake_loss = bin_cross(tf.zeros_like(fake_pred), fake_pred)

    return real_loss + fake_loss, real_loss, fake_loss


def generator_loss(fake_pred):
    """
    Define the cutom loss function for generator.

    It is a three-term loss:

    Lg = 0.5 * Lp + 0.006 * Lx + 0.01 * Ladv

    Lp = MSE Loss
    Lx = Perceptual Loss
    Ladv = Discriminator Loss

    Args:
        fake_pred (tf.Tensor): Scalar, output of the discriminator

    Returns:
        Generator three-term loss function output
    """
    bin_cross = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return bin_cross(tf.ones_like(fake_pred), fake_pred)
