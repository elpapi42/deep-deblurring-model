#!/usr/bin/python
# coding=utf-8

"""Setup and install the package and all the dependencies."""

import os
from setuptools import setup, find_packages

requirements_folder = os.path.join(os.path.dirname(__file__), 'requirements/')

with open(os.path.join(requirements_folder, 'production.txt')) as pro:
    INSTALL_REQUIRES = pro.read()

# This will be enable when project reach production and have a testing suit
#with open(os.path.join(requirements_folder, 'test.txt')) as test:
#    TEST_REQUIRES = test.read()

setup(
    author='Deep Learning Research Team',
    #author_email='',
    name='deep-deblurring',
    license='MIT',
    description='Image Deblurring using Deep Learning Architecture',
    version='1.0.0',
    url='',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': INSTALL_REQUIRES,
    },
    classifiers=[
        'Development Status :: Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
    ],
)
