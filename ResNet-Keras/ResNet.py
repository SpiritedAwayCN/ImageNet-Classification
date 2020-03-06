# 先用ResNet试一下水，我还在慢慢写orz

import keras
import argparse
import numpy as np

# set GPU memory 
if('tensorflow' == keras.backend.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser()

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

