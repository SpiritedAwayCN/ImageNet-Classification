# ImageNet Classification

AlexNet, ZFNet, ResNet, ResNetV2, SENet were implemented and some models trained on ILSVRC2012 and mini ImageNet dataset are available. 

## Accuracy

| model name      | dataset       | top-1 val err | top-5 val err | top-1 test err | top-5 test err |
| --------------- | ------------- | ------------- | ------------- | -------------- | -------------- |
| ResNetSE-50     | mini imageNet | 16.36%        | 4.80%         | 16.78%         | 4.78%          |
| ocd-ResNetV2-50 | ILSVRC 2012   | 21.95%        | 5.81%         | ---            | ---            |
| Average         | ILSVRC 2012   | 21.14%        | 5.50%         | ---            | ---            |

**Note:** Only the best models for each dataset are listed here, "Average" is predicted by the average score of two ResNetV2-50 models trained  independently on ILSVRC 2012  dataset. For error rates of all 13 provided models, please check ```test_single_image.ipynb```.

The weights files are provided here: [To Be Update]()

中国大陆地区可从这里下载 For friends in China Mainland: [网盘下载](https://disk.pku.edu.cn:443/link/1D00E5B1E1F7931ACFF5F9DD6DDA87DF)

It is strongly recommended to save the weights files in ```./h5``` directory, since test function will automatically load models there.

## Requirements

### For training:

* tensorflow >= 2.0.0
* opencv-python
* tqdm >= 4.42.0
* numpy

### For testing:

* tensorflow >= 2.0.0
* opencv-python
* numpy
* tqdm >= 4.42.0 (if applying seam carving)

## Models

All the architectures are from corresponding references, and may have been slightly adjusted. 

For example, "AlexNet-BN" replaced all Local Response Normalization layers to Batch Normalization. This scheme significantly reduces both the top-1 and top-5 error rates.

| model               | references                                                   |
| ------------------- | ------------------------------------------------------------ |
| AlexNet, AlexNet-BN | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) |
| ZFNet-BN            | [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) |
| ResNetV2            | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) |
| ResNetSE            | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) |

All the models above were trained on mini imageNet dataset and ResNetV2 was trained on ILSVRC 2012 as well. These models were implemented in ```models``` directory, and can used for your own training task.

## Dataset

* mini ImageNet - A subset of ImageNet, with 60,000 images in total contains 100 categories listed in ```./data/miniImageNet/labels_to_content.txt```.
* ILSVRC 2012 - The dataset used in ILSVRC 2012, contained 1,000 categories listed in ```./data/ILSVRC2012/labels_to_content.txt``` and a training set of 128,1167 images.

## Usage

### For testing single image

You can directly use the provided weights files to predict your own image. ```./test_single_image.ipynb``` is the user interface with detailed instructions and intuitive visualization. Check it for more information.

Meanwhile, ```./test_single_image.py``` is another predictor and able to compute average scores on two different models. You may need to modify the code before running, **deprecated if you don't know what the code is doing**.

### For training

The following is where the code needs to be modified:

* ```./constants.py``` - Modify the input size, batch size, epoch number, weight decay, number of classes, size of training set & validation set, numerical characteristics of dataset as your intensions.
* ```./utils/data.py``` - A file including training/validation images and its labels will be loaded during the training process (such as ```train_labels.txt``` and ```valid_labels.txt``` in ```data```'s subdirectory). The paths of these files and image root directory need to be defined there. Moreover, you can modify the data augmentation algorithm as you like.
* ```./train.py``` - Choose the training model implemented in ```models``` directory or your own. Set checkpoint file path, learning rate and its decay strategy.

Then, run these command to start your training:

```shell
$ python3 train.py
```

The default code will train ILSVRC2012 dataset on ResNetV2-50 model. For GTX 1080 Ti, it takes about 4 days to run 50 epochs.

### For testing 10-crop on test set

The following is where the code needs to be modified:

* ```./constants.py``` - Modify the input size, number of classes and numerical characteristics of trained dataset.
* ```./utils/data.py``` - Modify the paths of image root directory, and```val_labels.txt``` or ```test_labels.txt``` mentioned above.
* ```./test.py``` - Choose the model and set the path of weights file.

Then, run these command to start your testing, top-1 and top-5 accuracy will be reported:

```shell
$ python3 test.py
```

The default code will test ILSVRC2012 validation set of ```./h5/ocd-ResNetV2-50.h5``` model. 

## Limitation

All trainable variables were saved on GPU rather than CPU, and it will only use a single GPU to compute even if there are multiple GPUs available. This is because my computer has only one GPU, not able to debug or run multi-gpu training.

As the consequence, it's easy to have the problem that ```CUDA out of memory``` if the model is too deep and batch size is too large. Experiments shows that this will happen when using ResNetSE-50 model with batch size of 128. Try to decrease the batch size, simplify your model or use ```tf.float16``` to train if you encounter this problem.

## References

* The numerical characteristics of ILSVRC 2012 dataset and some implementation details of data input & data augmentation come from [ImageNet_ResNet_Tensorflow2.0](https://github.com/Apm5/ImageNet_ResNet_Tensorflow2.0) by [Apm5](https://github.com/Apm5). 

**Note:** Numerical characteristics of mini ImageNet can be extracted by ```./data/utils/image_cal.py```, it can also extract them from ILSVRC2012 dataset as well but costs plenty of time since the training set is too large.
