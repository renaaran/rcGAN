![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

rcGAN: Learning a generative model for arbitrary size image generation.<br>
[Renato B. Arantes](https://github.com/renaaran/),  [George Vogiatzis](http://george-vogiatzis.org//), and [Diego R. Faria](https://cs.aston.ac.uk/~fariad/)<br>
In ISVC 2020 (Oral).

## Installation

Clone this repo.
```bash
git clone https://github.com/renaaran/rcGAN.git
cd rcGAN/
```

This code requires PyTorch 1+ and python 3+. 

All models were trained on an NVIDIA GeForce RTX 2080 Ti.

## Training New Models

New models can be trained with the following commands.

```bash
python randomly_conditioned_dcgan.py --inputImagePath [path to source image] --outputFolder [output folder] --epochs [number of epochs] --gpu_id [gpu id]
```
The program generates intermediate samples on every epoch in the output folder as also checkpoints on every 50 epochs.

## Synthesising Images

After training the model on an image we are ready to synthesise samples.

```bash
python gan_image_quilting.py --inputImagePath [path to source image, same used to train the model]--outputFolder [output folder] --numberOfTiles [height] [widht] --n [number of images to synthesise] --modelPath [generator model path]
```

The example below generates five images, each having five blocks height and eight blocks width.

```bash
python gan_image_quilting.py --inputImagePath nature_sky.jpeg --outputFolder ./output/ --numberOfTiles 5 8 --n 5 --modelPath ./models/00399_09201_gen_model.1.18622.dic
```
