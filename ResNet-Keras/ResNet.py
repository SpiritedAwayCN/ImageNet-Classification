# 先用ResNet试一下水，我还在慢慢写orz

import keras
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

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
parser.add_argument('-n','--stack_n', type=int, default=3, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 3)')

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
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
    if epoch < 75:
        return 0.1
    elif epoch < 120:
        return 0.01
    return 0.001

def residual_block(x_in, filter_out, increase = False):
    stride = (2, 2) if increase else (1, 1)

    x1 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x_in))
    x2 = keras.layers.Conv2D(filters=filter_out,kernel_size=(3,3),strides=stride,padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(x1)
    x3 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x2))
    x4 = keras.layers.Conv2D(filters=filter_out,kernel_size=(3,3),strides=(1,1),padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(x3)
    if increase:
        res = keras.layers.Conv2D(filters=filter_out,kernel_size=(3,3),strides=stride,padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(x1)
        return keras.layers.add([x4, res])
    else:
        return keras.layers.add([x4, x_in])
        


def residual_network(img_input, num_classes, stack_n):
    x = keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(img_input)
    
    for _ in range(stack_n):
        x = residual_block(x, 16, False)
    
    x = residual_block(x, 32, True)
    for _ in range(stack_n):
        x = residual_block(x, 32, False)
    
    x = residual_block(x, 64, True)
    for _ in range(stack_n):
        x = residual_block(x, 64, False)
    
    x = keras.layers.Activation('relu')(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(num_classes ,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    
    return x


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
    output = residual_network(img_input,num_classes,stack_n)
    resnet = keras.models.Model(img_input, output)

    print(resnet.summary())

    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    cbks = [TensorBoard(log_dir='./resnet_{:d}_cifar10/'.format(layers), histogram_freq=0),
            LearningRateScheduler(scheduler),
            ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=5)]

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    resnet.save('resnet_{:d}_cifar10.h5'.format(layers))