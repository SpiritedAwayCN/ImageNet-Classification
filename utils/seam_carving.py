import cv2
import numpy as np
from tqdm import trange
import tensorflow as tf


def calc_energy(img):
    filter_du = tf.constant([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_du = tf.stack([filter_du] * 3, axis=2)

    filter_dv = tf.constant([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_dv = tf.stack([filter_dv] * 3, axis=2)
    r, c, v = tf.shape(img)
    img = tf.reshape(img, [1, r, c, v])
    filter_du = tf.reshape(filter_du, [3, 3, 3, 1])
    filter_dv = tf.reshape(filter_dv, [3, 3, 3, 1])

    img = tf.cast(img, dtype=tf.float32)
    x = tf.nn.conv2d(img, filter_du, strides=[1, 1, 1, 1], padding="SAME")
    y = tf.nn.conv2d(img, filter_dv, strides=[1, 1, 1, 1], padding="SAME")
    convolved = tf.math.abs(x + tf.math.abs(y))

    a, b, c, d = tf.shape(convolved)
    # 我们计算红，绿，蓝通道中的能量值之和
    energy_map = tf.reshape(convolved, [b, c])

    return energy_map


def crop_c(img):
    r, c, _ = tf.shape(img)

    for i in trange(c - 224):
        img = carve_column(img)

    return img


def crop_r(img):
    img = tf.image.rot90(img, 1)
    img = crop_c(img)
    img = tf.image.rot90(img, 3)
    return img


def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = tf.math.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = tf.stack([mask] * 3, axis=2)
    img = tf.reshape(img[mask],[r, c - 1, 3])
    return img.numpy()


def minimum_seam(img):
    r, c, _ = img.shape
    M = calc_energy(img)
    backtrack = np.zeros(tf.shape(M), dtype=np.int)

    M = M.numpy()

    for i in range(1, r):
        for j in range(0, c):
            # 处理图像的左侧边缘，确保我们不会索引-1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return tf.convert_to_tensor(M), tf.convert_to_tensor(backtrack)


def rescale_short_edge(image, size=None):
    if size is None:
        new_size = np.random.randint(256, 384)
    else:
        new_size = size
    height, weight, _ = np.shape(image)
    ratio = new_size / min(height, weight)
    return cv2.resize(image, (int(weight * ratio), int(height * ratio)))


def carve(img):
    r, c, _ = img.shape
    img = rescale_short_edge(img, 224)

    img = tf.convert_to_tensor(img)
    if r > c:
        out = crop_r(img)
    else:
        out = crop_c(img)
    return out

