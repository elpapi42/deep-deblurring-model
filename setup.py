#!/usr/bin/python
# coding=utf-8

"""Setup and install the package and all the dependencies."""

from setuptools import setup, find_packages

with open('requirements.txt') as pro:
    INSTALL_REQUIRES = pro.read().split('\n')

setup(
    author='Whitman Bohorquez, Mo Rebaie',
    author_email='whitman-2@hotmail.com',
    name='deblurrer',
    license='MIT',
    description='Image Deblurring using Deep Learning Architecture',
    version='1.0.0',
    url='',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        'Development Status :: Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
    ],
)
