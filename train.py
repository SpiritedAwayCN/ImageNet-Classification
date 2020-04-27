import tensorflow as tf
import os
import re
import numpy as np
from tqdm import tqdm, trange
from tensorflow import keras

import constants as c
from models.ResNetV2_50 import ResNet_v2_50
from utils.data import get_train_dataset, get_val_dataset
from utils.utils import l2_loss_of_model, correct_number
from test import test

@tf.function
def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        cross_entropy = keras.losses.categorical_crossentropy(labels, prediction, label_smoothing=c.label_smoothing)
        cross_entropy = tf.reduce_mean(cross_entropy)
        loss = cross_entropy + l2_loss_of_model(model)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

@tf.function
def warmup_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        cross_entropy = keras.losses.categorical_crossentropy(labels, prediction, label_smoothing=c.label_smoothing)
        cross_entropy = tf.reduce_mean(cross_entropy)
        loss = cross_entropy + l2_loss_of_model(model)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

def warmup(model, data_iter):
    learning_rate_schedules = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.0001,decay_steps=c.iterations_per_epoch,end_learning_rate=0.05)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)

    with trange(c.iterations_per_epoch, ncols=140) as t:
        for i in t:
            images, labels = data_iter.next()
            loss, prediction = warmup_step(model, images, labels, optimizer)
            correct_num = correct_number(labels, prediction)
            t.set_postfix_str('loss: {:.4f}, accurancy: {:.4f}'.format(loss, correct_num / images.shape[0]))

def train(model, data_iter, optimizer):
    with trange(c.iterations_per_epoch, ncols=140) as t:
        for i in t:
            images, labels = data_iter.next()
            loss, prediction = train_step(model, images, labels, optimizer)
            correct_num = correct_number(labels, prediction)
            
            t.set_postfix_str('loss: {:.4f}, accurancy: {:.4f}'.format(loss, correct_num / images.shape[0]))

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, decay1=15, decay2=30, lr=0.1):
        super(CustomSchedule, self).__init__()
        self.decay1 = decay1 * c.iterations_per_epoch
        self.decay2 = decay2 * c.iterations_per_epoch
        self.init_lr = lr

    @tf.function
    def __call__(self, step):
        if(step > self.decay2):
            return self.init_lr / 100
        elif(step > self.decay1):
            return self.init_lr / 10
        else:
            return self.init_lr

if __name__=='__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    # tf.keras.backend.set_floatx('float16')

    # comment below if applying miniImageNet
    c.num_class, c.mean, c.std, c.val_num, c.train_num = 1000, c.mean_o, c.std_o, 50000, 1281167

    model = ResNet_v2_50()
    model.build(input_shape=(None,) + c.input_shape)

    check_point_files = [int(re.findall(r"(\d+).h5",file)[0]) for file in os.listdir("./") if re.match(r'ResNetV2_50-\d+.h5$', file)]
    check_point_files.sort()
    init_epoch = 0 if len(check_point_files)==0 else check_point_files[-1]

    train_iter = get_train_dataset().__iter__()

    if init_epoch==0 and not os.path.isfile("ResNetV2_50-warmup.h5"):
        # warm up - 1 epoch
        warmup(model, train_iter)
        model.save_weights('ResNetV2_50-warmup.h5')
        test(model)
    elif init_epoch==0:
        model.load_weights('ResNetV2_50-warmup.h5')
    else:
        model.load_weights('ResNetV2_50-{:0>2}.h5'.format(init_epoch))
    
    learning_rate_schedules = keras.experimental.CosineDecay(initial_learning_rate=0.05,decay_steps=c.total_epoches * c.iterations_per_epoch, alpha=0.0001)
    # learning_rate_schedules = CustomSchedule()
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)

    # retore iteration step
    optimizer.iterations.assign(init_epoch * c.iterations_per_epoch)
    print(optimizer.iterations)

    for epoch in range(init_epoch, c.total_epoches):
        print("Epoch {:d}/{:d}".format(epoch + 1, c.total_epoches))
        train(model, train_iter, optimizer)
        model.save_weights('ResNetV2_50-{:0>2}.h5'.format(epoch + 1))
        test(model)