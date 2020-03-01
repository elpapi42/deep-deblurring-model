#!/usr/bin/python
# coding=utf-8

"""
Defines the cutom loss function for generator.

It is a three-term loss:

Lg = 0.5 * Lp + 0.006 * Lx + 0.01 * Ladv

Lp = MSE Loss
Lx = Perceptual Loss
Ladv = Discriminator Loss
"""
