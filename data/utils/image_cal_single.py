import numpy as np
import cv2
from tqdm import trange

file_name = ["train_label.txt", "test_label.txt", "val_label.txt"]
file_num = [50000, 25, 25]


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
        + line_split[0]).astype(np.float32)
    Image_tmp = rescale_short_edge(Image_tmp, 256)
    Image = Image_tmp.reshape(-1, 3)
    return Image


def calculate(Image, previous, current, EX, EXY):
    previous_ratio = previous / (previous + current)
    current_ratio = current / (previous + current)
    EX = np.mean(Image.T, axis=1) * current_ratio + previous_ratio * EX
    Image_sum = np.dot(Image.T, Image)
    EXY = Image_sum / (previous + current) + previous_ratio * EXY
    return EX, EXY


def get_expectation(path, num):
    with open(
            'E:\\Jupiter notebook\\Cifar10-100-Classification\\mini-imagenet\\'
            + path,
            'r',
            encoding="utf-8") as file_read:
        EX = np.zeros((3))
        EXY = np.zeros((3, 3))
        current_num = 0
        for j in trange(num):
            line = file_read.readline()
            Image = get_matrix(line)
            EX, EXY = calculate(Image, current_num,
                                np.array(Image).shape[0], EX, EXY)
            current_num += np.array(Image).shape[0]
    print(path)
    print("EX:")
    print(EX)
    print("EXY:")
    print(EXY)

if __name__ == '__main__':

    for i in range(3):
        get_expectation(file_name[i], file_num[i])
