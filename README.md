# withoutbg.com Background Removal Tool - Model Development

This repo is a simplified version of the model development framework for [withoutbg.com](https://withoutbg.com/).  
The web app is a free service. [Contact](https://withoutbg.com/contact/) for the API pricing.

## Architecture

The architecture is a UNet with a refiner. The backbone can be changed with a SOTA network like ResNet50.  
The refiner is sharpening the predicted alpha channel. It is a method mentioned in
the [Deep Image Matting](https://arxiv.org/abs/1703.03872) paper.

## Loss Function

The loss is a weighted average of the _compositional_ and _alpha prediction_ losses.  
Compositional loss is mentioned in the [Deep Image Matting](https://arxiv.org/abs/1703.03872) paper.  
Alternatively, an adversarial loss from the discriminator can be added to the weighted average. Check
the [AlphaGAN](https://arxiv.org/abs/1807.10088) paper for more information.

## Input Data

The input is a 4 channel input: RGB image (3 channels) and inverse depth map (1 channel).  
Depth Map: Because trimap is a human-in-the-loop solution, an inverse depth map is preferred. It is extracted by
using [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) model.  
Inputs are augmented with [Albumentations](https://albumentations.ai/) library.  
Input image is a composited image. To composite an image, a suitable background is chosen for the foreground. For
example, highway or parking lot backgrounds might be chosen as car backgrounds.  

## Conda Environment

To create the conda environment for the project:

```commandline
conda env create -f environment.yml
```