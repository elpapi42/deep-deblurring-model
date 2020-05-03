# Deep Deblurring Model
![Development Status](https://github.com/ElPapi42/deep-deblurring-model/workflows/Test%20Deblurrer/badge.svg)

Image Deblurring Deep Learning Model Built Over Tensorflow 2.X and Keras

The architeture was built taking the specifications from DeblurGAN-V2 Paper by Orest Kupyn and Team "[DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826)" as well as the Datasets recomended by the authors.

![Image](https://github.com/TAMU-VITA/DeblurGANv2/blob/master/doc_images/pipeline.jpg "architecture")

For training the model, we setup a training pipeline making the most out of Google Colab (We dont have budget for powerful local machines or Cloud Computing services). This repo contains the Jupyter Notebooks used as well as the Python Scripts for the different parts the the pipeline.

The Jupyter Notebook contains the Deployment Logic for easier interations for the Researchers. This model iterations are commited to [Deep Deblurring Serving](https://github.com/ElPapi42/deep-deblurring-serving) repository. From it, the Serving/Deployment of the model its done. Details on the serving repository.
