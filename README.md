# Interface for semantic editing in Unreal engine via python server
![](ezgif.com-optimize.gif)

This repository demonstrates the use of an existing ML model InterfaceGAN inside UE 5.1 for anonymous virtual identity creation and customization in Metafluxion. 

The file https://github.com/keertimanney/InterfaceGAN_UE5_PythonServer/blob/master/main.py is the python server which interacts with the ML model. It also handles httpps requests sent by user in game mode using Varest plugin in Unreal Engine to control and modify the attributes ( age, gender, seed ) and saves the generated image locally on the disk. This image is continuously updated in the widget displayed inside UE 5.1

# ML model used : InterFaceGAN - Interpreting the Latent Space of GANs for Semantic Face Editing

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)
![TensorFlow 1.12.2](https://img.shields.io/badge/tensorflow-1.12.2-green.svg?style=plastic)
![sklearn 0.21.2](https://img.shields.io/badge/sklearn-0.21.2-green.svg?style=plastic)

![image](./docs/assets/teaser.jpg)
**Figure:** *High-quality facial attributes editing results with InterFaceGAN.*

Specifically, InterFaceGAN is capable of turning an unconditionally trained face synthesis model to controllable GAN by interpreting the very first latent space and finding the hidden semantic subspaces.

[[Paper (CVPR)](https://arxiv.org/pdf/1907.10786.pdf)]
[[Paper (TPAMI)](https://arxiv.org/pdf/2005.09635.pdf)]
[[Project Page](https://genforce.github.io/interfacegan/)]
[[Demo](https://www.youtube.com/watch?v=uoftpl3Bj6w)]
[[Colab](https://colab.research.google.com/github/genforce/interfacegan/blob/master/docs/InterFaceGAN.ipynb)]


