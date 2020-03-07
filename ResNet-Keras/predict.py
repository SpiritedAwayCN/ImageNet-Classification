import keras
import argparse
import numpy as np

# set parameters via parser
parser = argparse.ArgumentParser()

parser.add_argument('-m','--model', type=str, default='./resnet_20_cifar10.h5', metavar='PATH',
                help='path of trained model')

args = parser.parse_args()

model = keras.models.load_model(args.model)

print(model.summary())