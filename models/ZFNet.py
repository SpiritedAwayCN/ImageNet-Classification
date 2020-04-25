'''
超参数：
    learning rate
    weight decay
Dense: 全相联
BatchNormailization: (x-均值)/sqrt{标准差^2 + epsilon} * a + b
'''
import tensorflow as tf
import constants as c
from tensorflow import keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout

# num_classes        = 100
# img_rows, img_cols, img_channels = 224, 224, 3
# batch_size         = 128
# epochs             = 100
# iterations         = 50000 // batch_size + 1
# weight_decay       = 1e-4


# def alexNetInference(img_input, num_classes):
#     conv1 = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding="same")(img_input)
#     conv1 = Activation('relu')(conv1)
#     norm1 = tf.nn.lrn(conv1, 5, 2.0, alpha=1e-4, beta=0.75)
#     pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(norm1)
    
#     conv2 = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same")(pool1)
#     conv2 = Activation('relu')(conv2)
#     norm2 = tf.nn.lrn(conv2, 5, 2.0, alpha=1e-4, beta=0.75)
#     pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(norm2)
    
#     conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same")(pool2)
#     conv3 = Activation('relu')(conv3)
    
#     conv4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same")(conv3)
#     conv4 = Activation('relu')(conv4)
    
#     conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(conv4)
#     conv5 = Activation('relu')(conv5)
#     pool3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(conv5)

#     flatten = Flatten()(pool3)
#     dense1 = Dense(4096, activation='relu')(flatten)
#     dropout1 = Dropout(rate=0.5)(dense1)
#     dense2 = Dense(4096, activation='relu')(dropout1)
#     dropout2 = Dropout(rate=0.5)(dense2)
#     return Dense(num_classes, activation='softmax')(dropout2)

class ZFNet(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(ZFNet, self).__init__(**kwargs)
        # 2013年冠军是参数优化后的AlexNet(ZFnet)，第一层是7*7，步长2*2，其余层不变
        # self.conv1 = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', padding="same")
        self.conv1 = Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), activation='relu', padding="same") # ZFNet
        self.pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")
        self.conv2 = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation='relu', padding="same")
        self.pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")

        self.conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.pool3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")

        self.bn_0 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.fc_1 = Dense(4096, activation='relu')
        self.fc_2 = Dense(4096, activation='relu')
        self.fc_3 = Dense(c.num_class, activation='softmax')
    
    def call(self, inputs, training):
        res = self.conv1(inputs)
        # res = tf.nn.lrn(res, 5, 2.0, alpha=1e-4, beta=0.75)
        res = self.bn_0(res, training=training)
        res = self.pool1(res)

        res = self.conv2(res)
        # res = tf.nn.lrn(res, 5, 2.0, alpha=1e-4, beta=0.75)
        res = self.bn_1(res, training=training)
        res = self.pool2(res)

        res = self.conv3(res)
        res = self.conv4(res)
        res = self.conv5(res)
        res = self.pool3(res)

        res = Flatten()(res)
        res = self.fc_1(res)
        if training:
            res = Dropout(rate=0.5)(res)
        res = self.fc_2(res)
        if training:
            res = Dropout(rate=0.5)(res)
        outputs = self.fc_3(res)

        return outputs



if __name__ == '__main__':
    # img_input = keras.layers.Input(shape=(img_rows,img_cols,img_channels))
    # output = alexNetInference(img_input,num_classes)
    # alexnet = keras.models.Model(img_input, output)
    # alexnet.save('alexNet_cifar10.h5')
    model = ZFNet()
    model.build((None, ) + c.input_shape)

    cnt1 = cnt2 = 0
    for v in model.trainable_variables:
        print(v.name)
        cnt1 += 1
        if 'kernel' in v.name:
            cnt2 += 1
    print(cnt1, cnt2)
    print(model.summary())
    