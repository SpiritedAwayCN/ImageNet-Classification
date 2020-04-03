import keras
import random
import numpy as np
from numpy import linalg as la


def PCA_svd(x_train):
    x_shape = np.array(x_train).shape
    x_svd = np.zeros((x_shape[3], x_shape[0] * x_shape[1] * x_shape[2]))
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    for i in range(x_shape[3]):
        x_svd[i] = x_train[:, :, :, i].reshape(
            [x_shape[0] * x_shape[1] * x_shape[2]])
    x_cov = np.cov(x_svd)
    u, sigma, vt = la.svd(x_cov)
    return sigma, vt


def apart(x_test, y_test):
    random.seed(len(x_test))
    np.random.seed(6662860)
    np.random.shuffle(x_test)
    np.random.seed(6662860)
    np.random.shuffle(y_test)
    x_val = x_test[0:int(len(x_test) / 2), :, :, :]
    x_test = x_test[int(len(x_test) / 2):len(x_test), :, :, :]
    y_val = y_test[0:int(len(y_test) / 2), :]
    y_test = y_test[int(len(y_test) / 2):len(y_test), :]
    return x_val, y_val, x_test, y_test

