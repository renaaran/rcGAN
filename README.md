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
python randomly_conditioned_dcgan.py --inputImagePath=[path to source image] --outputFolder=[output folder] --epochs=[number of epochs] --gpu_id=[gpu id]
```
The program generates intermediate samples on every epoch in the output folder as also checkpoints on every 50 epochs.
