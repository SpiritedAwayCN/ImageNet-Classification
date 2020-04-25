import tensorflow as tf
import numpy as np
from tensorflow import keras
from tqdm import trange

from utils.data import get_val_dataset, get_predict_dataset
from utils.utils import correct_number
import constants as c

@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    cross_entropy = keras.losses.categorical_crossentropy(labels, prediction, label_smoothing=c.label_smoothing)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy, prediction

def test(model):
    data_iterator = get_val_dataset().__iter__()
    total_correct_num = 0

    print("testing...")

    with trange(c.test_iterations, ncols=140) as t:
        for i in t:
            images, labels = data_iterator.next()

            cross_entropy, prediction = test_step(model, images, labels)
            correct_num = correct_number(labels, prediction)

            total_correct_num += correct_num
            t.set_postfix_str('ce: {:.4f}, accuracy: {:.4f}'.format(cross_entropy, correct_num / images.shape[0]))

    print('accuracy {:.4f}'.format(total_correct_num / c.val_num))

@tf.function
def test_multicrop_step(model, images):
    prediction = model(images, training=False)
    prediction = tf.reduce_mean(prediction, axis=0) # 平均
    return prediction

def test_multicrop(model, top_k=1):
    data_iterator = get_predict_dataset().__iter__()

    total_correct_num = 0
    total_correct_top1 = 0

    print("testing multicrop (top-{:d})...".format(top_k))

    with trange(c.val_num, ncols=100) as t:
        for i in t:
            images, labels = data_iterator.next()
            prediction = test_multicrop_step(model, images)

            total_correct_top1 += 1 if np.argmax(labels) == np.argmax(prediction) else 0
            total_correct_num += 1 if np.argmax(labels) in np.argpartition(prediction, -top_k)[-top_k:] else 0
            t.set_postfix_str('correct_num: {:d}'.format(total_correct_num))

    print('top-1 accuracy {:.4f}, top-{:d} accuracy {:.4f}.'.format(total_correct_top1 / c.val_num, top_k, total_correct_num / c.val_num))

if __name__=='__main__':
    from models.AlexNet_bn import AlexNet_BN
    # from models.ResNetV2_18 import ResNet_v2_18

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    model = AlexNet_BN()
    model.build(input_shape=(None,) + c.input_shape)

    model.load_weights('./h5/AlexNet-BN.h5')
    
    test_multicrop(model, top_k=5)