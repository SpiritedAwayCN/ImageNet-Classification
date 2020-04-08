import tensorflow as tf
import numpy as np
import cv2
import json
import os

import constants as c
from ResNetV2_18 import ResNet_v2_18

# 内测用，少一点必要的别的python更好

model_path = './resnetV2-18-50.h5'
image_path = './image.jpg'
show_top_k = 10

@tf.function
def test_multicrop_step(model, images):
    prediction = model(images, training=False)
    prediction = tf.reduce_mean(prediction, axis=0) # 平均
    return prediction

def rescale_short_edge(image, new_size = None):
    if new_size is None:
        new_size = np.random.randint(256, 288)
    height, weight, _ = np.shape(image)
    ratio = new_size / min(height, weight)
    return cv2.resize(image, (int(weight * ratio), int(height * ratio)))

def load_image_multicrop(path):
    image = cv2.imread(path).astype(np.float32)
    image = rescale_short_edge(image, new_size=256)

    height, width, _ = np.shape(image)
    input_height, input_width, _ = c.input_shape
    center_crop_x = (width - input_width) // 2
    center_crop_y = (height - input_height) // 2

    images = []
    images.append(image[:input_height, :input_width, :])  # left top
    images.append(image[:input_height, -input_width:, :])  # right top
    images.append(image[-input_height:, :input_width, :])  # left bottom
    images.append(image[-input_height:, -input_width:, :])  # right bottom
    images.append(image[center_crop_y: center_crop_y + input_height, center_crop_x: center_crop_x + input_width, :])

    image = cv2.flip(image, 1)
    images.append(image[:input_height, :input_width, :])  # left top
    images.append(image[:input_height, -input_width:, :])  # right top
    images.append(image[-input_height:, :input_width, :])  # left bottom
    images.append(image[-input_height:, -input_width:, :])  # right bottom
    images.append(image[center_crop_y: center_crop_y + input_height, center_crop_x: center_crop_x + input_width, :])

    image = np.array(images, dtype=np.float32)

    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]
    
    return image


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

print("loading model...")
model = ResNet_v2_18()
model.build(input_shape=(None,) + c.input_shape)
model.load_weights(model_path)

print("loading image...")
image = load_image_multicrop(image_path)
print("predicting image...")
prediction = test_multicrop_step(model, image)

index_list = np.argpartition(prediction, -show_top_k)[-show_top_k:]

index_list = sorted(index_list, key=lambda x: -prediction[x].numpy())

with open("data/mini-labels_to_contents.json") as f:
    label_dict = json.loads(f.readline())

    print('-' * 40)
    print("Predictions top-{:d}:".format(show_top_k))
    for lable in index_list:
        print("{:.4f} \t {}".format(prediction[lable].numpy(), label_dict[str(lable)]))
    print('-' * 40)