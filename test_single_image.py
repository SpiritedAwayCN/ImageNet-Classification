import tensorflow as tf
import numpy as np
import cv2
import json
import os

import constants as c
from models.ResNetV2_50 import ResNet_v2_50
from utils.data import rescale_short_edge
from utils.utils import crop_ten
from utils import seam_carving

model_path_1 = './h5/ocd-ResNetV2-50.h5'
model_path_2 = './h5/oResNetV2-50.h5'
image_path = './image.jpg'
show_top_k = 10
apply_seam_carving = False

# comment below if applying miniImageNet
c.num_class, c.mean, c.std = 1000, c.mean_o, c.std_o


def load_image_multicrop(path):
    image = cv2.imread(path).astype(np.float32)
    image = rescale_short_edge(image, new_size=256)

    images = crop_ten(image)

    if apply_seam_carving:
        images.append(seam_carving.carve(image))

    image = np.array(images, dtype=np.float32)

    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]

    return image


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    device=physical_devices[0], enable=True)

print("loading model...")
model_1 = ResNet_v2_50()
model_1.build(input_shape=(None,) + c.input_shape)
model_1.load_weights(model_path_1)

model_2 = ResNet_v2_50()
model_2.build(input_shape=(None,) + c.input_shape)
model_2.load_weights(model_path_2)

print("loading image...")
image = load_image_multicrop(image_path)
print("predicting image...")
prediction_1 = model_1(image, training=False)
prediction_1 = tf.reduce_mean(prediction_1, axis=0)

prediction_2 = model_2(image, training=False)
prediction_2 = tf.reduce_mean(prediction_2, axis=0)

prediction = tf.add(prediction_1, prediction_2)/2

index_list = np.argpartition(prediction, -show_top_k)[-show_top_k:]

index_list = sorted(index_list, key=lambda x: -prediction[x].numpy())

with open("./data/ILSVRC2012/labels_to_content.txt") as f:
    label_dict = f.read().splitlines()

    print('-' * 40)
    print("Predictions top-{:d}:".format(show_top_k))
    for lable in index_list:
        print("{:.4f} \t {}".format(prediction[lable], label_dict[lable]))
    print('-' * 40)
