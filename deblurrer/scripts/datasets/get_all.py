#!/usr/bin/python
# coding=utf-8

"""
Run all the available download scripts.

"""

from deblurrer.scripts.datasets import kaggle_blur


def run():
    """
    Run all the datasets download scripts.

    """
    kaggle_blur.run()


if (__name__ == '__main__'):
    run()
