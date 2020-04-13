import sys
import cv2
from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve


def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(
        convolve(img, filter_dv))

    # 我们计算红，绿，蓝通道中的能量值之和
    energy_map = convolved.sum(axis=2)

    return energy_map


def crop_c(img):
    r, c, _ = img.shape

    for i in trange(c - 224):
        img = carve_column(img)

    return img


def crop_r(img):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img)
    img = np.rot90(img, 3, (0, 1))
    return img


def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img


def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

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

    return M, backtrack


def rescale_short_edge(image, size=None):
    if size is None:
        new_size = np.random.randint(256, 384)
    else:
        new_size = size
    height, weight, _ = np.shape(image)
    ratio = new_size / min(height, weight)
    return cv2.resize(image, (int(weight * ratio), int(height * ratio)))


def seam_carving(img):
    r, c, _ = img.shape
    img = rescale_short_edge(img, 224)
    if r > c:
        out = crop_r(img)
    else:
        out = crop_c(img)
    return out

