import tensorflow as tf
from tensorflow.python import keras
import tensorflow_datasets as tfds
import numpy as np
import os
import cv2
import constants as c

def load_list(list_path, image_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_path, line[0]))
            labels.append(int(line[1]))
    return images, labels

def rescale_short_edge(image, size = None):
    if size is None:
        new_size = np.random.randint(256, 384)
    height, weight, _ = np.shape(image)
    ratio = new_size / min(height, weight)
    return cv2.resize(image, (int(weight * ratio), int(height * ratio)))

def augment(image):
    # 长边伸缩
    height, weight, _ = np.shape(image)
    ratio = np.random.uniform(0.8, 1.25)
    if height >= weight:
        new_size = (weight, int(height * ratio)) # 反的
    else:
        new_size = (int(weight * ratio), height)
    image = cv2.resize(image, new_size)

    # 短边变为指定大小
    image = rescale_short_edge(image)

    # 随机裁剪
    height, weight, _ = np.shape(image)
    crop_x = np.random.randint(height - c.input_shape[0])
    crop_y = np.random.randint(weight - c.input_shape[1])
    image = image[crop_x:crop_x + c.input_shape[0], crop_y:crop_y + c.input_shape[1]]

    # 随机水平翻转
    if(np.random.rand() < 0.5):
        image = cv2.flip(image, 1)

    # hsv上的偏移
    offset_h = np.random.uniform(-36, 36) #这个不能太过分
    offset_s = np.random.uniform(0.6, 1.4)
    offset_v = np.random.uniform(0.6, 1.4)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = (image_hsv[:, :, 0] + offset_h) % 360.
    image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * offset_s, 1.)
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * offset_v, 255.)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # pca噪音 复用了两个变量
    offset_h = np.random.normal(0, 0.1, size=(3,))
    offset_s = np.dot(c.eigvec * offset_h, c.eigval)
    image = np.maximum(np.minimum(image + offset_s, 255.), 0.)

    return image

def load_image(path, labels, augments=False):
    image = cv2.imread(path.numpy().decode()).astype(np.float32)

    if augments:
        image = augment(image)
    else:
        rescale_short_edge(image, size=256)
        # TODO 裁剪中心224x224

    
    # 可视化请注释以下部分
    # for i in range(3):
    #     image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]

    label = keras.utils.to_categorical(labels, 1000)
    return image, label

def _parser(path, labels):
    return tf.py_function(load_image, inp=[path, labels, False], Tout=[tf.float32, tf.float32])

def _parser_with_augment(path, labels):
    return tf.py_function(load_image, inp=[path, labels, True], Tout=[tf.float32, tf.float32])

def train_iteration(list_path="train_label.txt"):
    images, labels = load_list(list_path, "E:\\Programming projects\\ILSVRC2012\\train")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images)).repeat()
    dataset = dataset.map(_parser_with_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(128)
    return dataset.make_one_shot_iterator()

if __name__=='__main__':
    train_iter = train_iteration()

    with tf.Session() as sess:
        image, labels = sess.run(train_iter.get_next())

        print(np.shape(image))
        for i in range(10):
            cv2.imshow('show', image[i].astype(np.uint8))
            cv2.waitKey(0)
        cv2.destroyAllWindows()