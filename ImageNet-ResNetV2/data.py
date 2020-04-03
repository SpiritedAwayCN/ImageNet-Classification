import tensorflow as tf
from tensorflow.python import keras
import tensorflow_datasets as tfds
import numpy as np
import os
import cv2

def load_list(list_path, image_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_path, line[0]))
            labels.append(int(line[1]))
    return images, labels

def augment(image):
    # TODO 各种增强
    return image

def load_image(path, labels, augments=False):
    image = cv2.imread(path.numpy().decode()).astype(np.float32)

    if augments:
        image = augment(image)
    else:
        None # TODO - crop the image to input shape
    
    # TODO 减去平均，除以标准差

    label = keras.utils.to_categorical(labels, 1000)
    return image, label

def _parser(path, labels):
    return tf.py_function(load_image, inp=[path, labels, False], Tout=[tf.float32, tf.float32])

def _parser_with_augment(path, label):
    return tf.py_function(load_image, inp=[path, labels, True], Tout=[tf.float32, tf.float32])

def train_iteration(list_path="train_label.txt"):
    images, labels = load_list(list_path, "E:\\Programming projects\\ILSVRC2012\\train")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.map(_parser_with_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(128)
    return dataset.make_one_shot_iterator()

if __name__=='__main__':
    train_iter = train_iteration()

    with tf.Session() as sess:
        image, labels = sess.run(train_iter.get_next())

        print(image)
        cv2.imshow('show', image.astype(np.uint8))
        cv2.waitKey(0)