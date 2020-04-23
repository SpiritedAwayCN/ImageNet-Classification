import tensorflow as tf
from tensorflow import keras
import constants as c

def l2_loss_of_model(model):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * c.weight_decay

def correct_number(labels, prediction):
    correct_num = tf.equal(tf.argmax(labels, -1), tf.argmax(prediction, -1))
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num