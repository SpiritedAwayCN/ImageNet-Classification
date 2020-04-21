import tensorflow as tf
import constants as c
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dense, Activation, AvgPool2D, MaxPooling2D

class BottleneckBlock(tf.keras.layers.Layer):
    # reference: https://keras.io/layers/writing-your-own-keras-layers/
    def __init__(self, filters=None, strides=(1, 1), projection=False, **kwargs):
        # self.filters = filters
        self.strides = strides
        self.projection = projection
        if projection or strides != (1, 1):
            self.shortcut = Conv2D(filters * 4, (1, 1), padding='same',use_bias=False)
            if not projection:
                self.avgpool = AvgPool2D((2, 2), strides=(2, 2), padding='same')

        self.conv_0 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.conv_2 = Conv2D(filters * 4, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.activation0 = Activation('relu')
        self.activation1 = Activation('relu')
        self.activation2 = Activation('relu')
        super(BottleneckBlock, self).__init__(**kwargs)
    
    def call(self, inputs, training):
        res = self.bn_0(inputs, training=training)

        res = self.activation0(res)
        
        if self.projection:
            shortcut = self.shortcut(res)
        elif self.strides != (1, 1):
            shortcut = self.avgpool(res) # 长宽除以2
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = inputs

        res = self.conv_0(res)

        res = self.bn_1(res, training=training)
        res = self.activation1(res)
        res = self.conv_1(res)

        res = self.bn_2(res, training=training)
        res = self.activation2(res)
        res = self.conv_2(res)
        output = res + shortcut
        return output
    
    # 不写貌似也没啥问题
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[1]/self.strides[0], input_shape[2]/self.strides[1], input_shape[3])

class ResNet_v2_50(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(ResNet_v2_50, self).__init__(**kwargs)

        self.conv0 = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)
        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        self.block_collector = []
        block_num = 3, 4, 6, 3
        filters_num = 64, 128, 256, 512
        for i in range(1, 5):
            if i == 1:
                self.block_collector.append(BottleneckBlock(filters_num[i-1], projection=True, name='conv1_0'))
            else:
                self.block_collector.append(BottleneckBlock(filters_num[i-1], strides=(2, 2), name='conv{}_0'.format(i)))

            for j in range(1, block_num[i-1]):
                self.block_collector.append(BottleneckBlock(filters_num[i-1], name='conv{}_{}'.format(i, j)))

        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)
        self.activation = Activation('relu')
        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(c.num_class, name='fully_connected', activation='softmax', use_bias=False)

    def call(self, inputs, training):
        net = self.conv0(inputs)
        # print('input', inputs.shape)
        # print('conv0', net.shape)
        net = self.maxpool(net)
        # print('max-pooling', net.shape)

        for block in self.block_collector:
            net = block(net, training)
            # print(block.name, net.shape)
        net = self.bn(net, training)
        net = self.activation(net)

        net = self.global_average_pooling(net)
        # print('global average-pooling', net.shape)
        net = self.fc(net)
        # print('fully connected', net.shape)

        return net


if __name__=='__main__':
    model = ResNet_v2_50()
    model.build((None, ) + c.input_shape)

    cnt1 = cnt2 = 0
    for v in model.trainable_variables:
        print(v.name)
        cnt1 += 1
        if 'kernel' in v.name:
            cnt2 += 1
    print(cnt1, cnt2)
    print(model.summary())