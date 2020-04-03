# 先用AlexNet试一下水，我还在慢慢写orz

from tensorflow.python import keras
import argparse
import numpy as np
from get_checkpoint import get_checkpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense

# set GPU memory 
if('tensorflow' == keras.backend.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser()

parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=160, metavar='NUMBER',
                help='epochs(default: 160)')

args = parser.parse_args()

num_classes        = 10
img_rows, img_cols, img_channels = 32, 32, 3
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # 这段拿的网上的
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 60:
        return 0.1
    elif epoch < 110:
        return 0.01
    return 0.001

def alexNetInference(img_input, num_classes):
    conv1 = Conv2D(filters=48, kernel_size=(5,5), strides=(1,1), padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(img_input)

    norm1 = Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(norm1)

    conv2 = Conv2D(filters=72, kernel_size=(5,5), strides=(1,1), padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(pool1)
    norm2 = Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv2))
    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(norm2)

    conv3 = Conv2D(filters=112, kernel_size=(3,3), strides=(2, 2), padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(pool2)
    norm3 = Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv3))

    conv4 = Conv2D(filters=112, kernel_size=(3,3), strides=(1, 1), padding="same",
                kernel_initializer="he_normal", activation="relu",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(norm3)
    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv4)

    flatten = Flatten()(pool3)
    dense1 = Dense(256, activation='relu',kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(flatten)
    return Dense(num_classes ,activation='softmax',kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(dense1)




if __name__ == '__main__':
    # 这段大部分也是网上的
    print("== LOADING DATA... ==")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = keras.layers.Input(shape=(img_rows,img_cols,img_channels))
    output = alexNetInference(img_input,num_classes)
    alexnet = keras.models.Model(img_input, output)

    print(alexnet.summary())

    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    alexnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # get check point
    checkpoint_file, current_epoch = get_checkpoint()
    if not checkpoint_file is None:
        print("Founded checkpoint file {}".format(checkpoint_file))
        alexnet.load_weights(checkpoint_file)
        

    cbks = [TensorBoard(log_dir='./alexNet_cifar10/', histogram_freq=0),
            LearningRateScheduler(scheduler),
            ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)]

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    alexnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test),
                         initial_epoch=current_epoch)
    alexnet.save('alexNet_cifar10.h5')