import tensorflow as tf
import os
import numpy as np
import constants as c
from tqdm import tqdm
from tensorflow import keras
from ResNetV2_50 import inference
from data import get_train_dataset, get_val_dataset
from utils import l2_loss_of_model, correct_number

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

def warmup(model, data_iter):
    learning_rate_schedules = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.0001,
                                                                   decay_steps=c.iterations_per_epoch,
                                                                   end_learning_rate=0.05)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)

    for i in tqdm(range(c.iterations_per_epoch)):
        images, labels = data_iter.next()
        loss, prediction = train_step(model, images, labels, optimizer)
        correct_num = correct_number(labels, prediction)
        print('loss: {:4f}, accurancy: {:4f}'.format(loss, correct_num / images.shape[0]))



if __name__=='__main__':
    # gpu config
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float16')

    img_input = keras.layers.Input(shape=c.input_shape)
    output = inference(img_input, training=True)
    model = keras.models.Model(img_input, output)

    
    train_iter = get_train_dataset().__iter__()
    # val_dataset = get_val_dataset()

    if not os.path.isfile("resnetV2-50-warmup.h5"):
        # warm up - 1 epoch
        warmup(model, train_iter)

        model.save('resnetV2-50-warmup.h5')