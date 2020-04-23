import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import constants as c

def l2_loss_of_model(model):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * c.weight_decay

def correct_number(labels, prediction):
    correct_num = tf.equal(tf.argmax(labels, -1), tf.argmax(prediction, -1))
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num

def crop_ten(image):
    height, width, _ = np.shape(image)
    input_height, input_width, _ = c.input_shape
    center_crop_x = (width - input_width) // 2
    center_crop_y = (height - input_height) // 2

    images = []
    images.append(image[:input_height, :input_width, :])  
    images.append(image[:input_height, -input_width:, :])
    images.append(image[-input_height:, :input_width, :])
    images.append(image[-input_height:, -input_width:, :])
    images.append(image[center_crop_y: center_crop_y + input_height, center_crop_x: center_crop_x + input_width, :])

    image = cv2.flip(image, 1)
    images.append(image[:input_height, :input_width, :])  
    images.append(image[:input_height, -input_width:, :])
    images.append(image[-input_height:, :input_width, :])
    images.append(image[-input_height:, -input_width:, :])
    images.append(image[center_crop_y: center_crop_y + input_height, center_crop_x: center_crop_x + input_width, :])

    return images