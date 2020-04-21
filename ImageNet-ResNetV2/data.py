import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import constants as c
import seam_carving

def load_list(list_path, image_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_path, line[0]))
            labels.append(int(line[1]))
    return images, labels

def rescale_short_edge(image, new_size = None):
    if new_size is None:
        new_size = np.random.randint(256, 288)
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
    offset_h = np.random.uniform(-18, 18) #这个不能太过分
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
        image = rescale_short_edge(image, new_size=256)
        height, width, _ = np.shape(image)
        input_height, input_width, _ = c.input_shape
        crop_x = (width - input_width) // 2
        crop_y = (height - input_height) // 2
        image = image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

    
    # 可视化请注释以下部分
    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]

    label = keras.utils.to_categorical(labels, c.num_class)
    return image, label

def load_image_multicrop(path, labels, seam_carv=True):
    image = cv2.imread(path.numpy().decode()).astype(np.float32)
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

    if seam_carv == True:
        images.append(seam_carving.carve(image))
    
    image = np.array(images, dtype=np.float32)

    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]
    label = keras.utils.to_categorical(labels, c.num_class)
    return image, label

def get_train_dataset(list_path="train_label.txt"):
    images, labels = load_list(list_path, "./ILSVRC2012/train")
    # images, labels = load_list(list_path, "E:\\Programming projects\\ILSVRC2012\\mini-imagenet\\train")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images)).repeat()
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, True], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    return dataset

def get_val_dataset(list_path="val_label.txt"):
    images, labels = load_list(list_path, "./ILSVRC2012/val")
    # images, labels = load_list(list_path, "E:\\Programming projects\\ILSVRC2012\\mini-imagenet\\val")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    return dataset

def get_predict_dataset(list_path="E:\Programming Projects\ILSVRC2012\mini-imagenet\\val_label - 副本.txt"):
    # images, labels = load_list(list_path, "./ILSVRC2012/val")
    images, labels = load_list(list_path, "E:\\Programming projects\\ILSVRC2012\\mini-imagenet\\val")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_multicrop, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__=='__main__':
    train_iter = get_val_dataset().__iter__()

    image, labels =train_iter.next()
    print(np.shape(image), np.shape(labels))

    for i in range(10):
        cv2.imshow('show', image[i].numpy().astype(np.uint8))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
