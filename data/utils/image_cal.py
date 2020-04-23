import numpy as np
import cv2
from tqdm import trange
import time

EX = np.zeros((3))
EXY = np.zeros((3))
EX2 = np.zeros((3))
file_name = ["test_label.txt", "train_label.txt", "val_label.txt"]
file_num = [25, 250, 25]


def rescale_short_edge(image, size=None):
    if size is None:
        new_size = np.random.randint(256, 384)
    else:
        new_size = size
    height, weight, _ = np.shape(image)
    ratio = new_size / min(height, weight)
    return cv2.resize(image, (int(weight * ratio), int(height * ratio)))


def get_matrix(line):
    line_split = line.split()
    Image_tmp = cv2.imread(
        'E:\\Jupiter notebook\\Cifar10-100-Classification\\mini-imagenet\\images\\'
        + line_split[0])
    Image_tmp = rescale_short_edge(Image_tmp, 256)
    Image_shape = np.array(Image_tmp).shape
    Image = np.zeros((3, Image_shape[0] * Image_shape[1]))
    for i in range(3):
        Image[i] = Image_tmp[:, :, i].reshape(
            [Image_shape[0] * Image_shape[1]])
    return Image


def calculate(Image, previous, current):
    previous_ratio = previous / (previous + current)
    current_ratio = current / (previous + current)
    for i in range(3):
        EX[i] = np.mean(Image[i]) * current_ratio + previous_ratio * EX[i]
    Image_cov = np.cov(Image)
    EXY[0] = Image_cov[0][1] * current_ratio + previous_ratio * EXY[0]
    EXY[1] = Image_cov[0][2] * current_ratio + previous_ratio * EXY[1]
    EXY[2] = Image_cov[1][2] * current_ratio + previous_ratio * EXY[2]
    for i in range(3):
        EX2[i] = Image_cov[i][i] * current_ratio + previous_ratio * EX2[i]


def get_expectation(path, num):
    with open(
            'E:\\Jupiter notebook\\Cifar10-100-Classification\\mini-imagenet\\'
            + path,
            'r',
            encoding="utf-8") as file_read:
        current_num = 0
        for j in trange(num):
            line = file_read.readline()
            Image = get_matrix(line)
            for k in range(199):
                line = file_read.readline()
                Image = np.hstack((Image, get_matrix(line)))

            calculate(Image, current_num, np.array(Image).shape[1])
            current_num += np.array(Image).shape[1]
    print(path)
    print("EX:")
    print(EX)
    print("EXY:")
    print(EXY)
    print("EX2:")
    print(EX2)


if __name__ == '__main__':
    for i in range(3):
        EX = np.zeros((3))
        EXY = np.zeros((3))
        EX2 = np.zeros((3))
        get_expectation(file_name[i], file_num[i])
