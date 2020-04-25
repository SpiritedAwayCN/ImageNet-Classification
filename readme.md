# ImageNet Classification

AlexNet, ZFNet, ResNet, ResNetV2, SENet were implemented and some models trained on ILSVRC2012 and mini ImageNet dataset are available. 

### Accuracy

| model name      | dataset       | top-1 val err | top-5 val err | top-1 test err | top-1 test err |
| --------------- | ------------- | ------------- | ------------- | -------------- | -------------- |
| ResNetSE-50     | mini imageNet | 16.36         | 4.80          | 16.78          | 4.78           |
| ocd-ResNetV2-50 | ILSVRC 2012   | 21.95         | 5.81          | ---            | ---            |

**Note:** Only the best models for each dataset are listed here. For all 13 provided models, please check ```test_single_image.ipynb```.

The weights files are provided here: [网址]()
It is strongly recommended to save the file in the ```./h5``` directory, since test function will search models there.

## Requirements

For training:

* tensorflow >= 2.0.0
* opencv-python
* tqdm >= 4.42.0
* numpy

For testing:

* tensorflow >= 2.0.0
* opencv-python
* numpy
* scipy (if applying seam carving)
* tqdm >= 4.42.0 (if applying seam carving)

## Dataset



