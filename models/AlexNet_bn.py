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

class AlexNet_BN(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(AlexNet_BN, self).__init__(**kwargs)
        # 2013年冠军是参数优化后的AlexNet(ZFnet)，第一层是7*7，步长2*2，其余层不变
        self.conv1 = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', padding="same")
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")
        self.conv2 = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation='relu', padding="same")
        self.bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")

        self.conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")
        self.pool3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")

        self.fc_1 = Dense(4096, activation='relu')
        self.fc_2 = Dense(4096, activation='relu')
        self.fc_3 = Dense(c.num_class, activation='softmax')
    
    def call(self, inputs, training):
        res = self.conv1(inputs)
        res = self.bn_1(res, training=training)
        res = self.pool1(res)

        res = self.conv2(res)
        res = self.bn_2(res, training=training)
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
    model = AlexNet()
    model.build((None, ) + c.input_shape)

    cnt1 = cnt2 = 0
    for v in model.trainable_variables:
        print(v.name)
        cnt1 += 1
        if 'kernel' in v.name:
            cnt2 += 1
    print(cnt1, cnt2)
    print(model.summary())
    