import tensorflow as tf
import keras
from keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dense, Activation, AvgPool2D, MaxPool2D, Input
from keras.regularizers import l2

_weight_decay = 1e-4
_category_num = 1000

# 我放弃了
# class BottleneckBlock(keras.layers.Layer):
#     # reference: https://keras.io/layers/writing-your-own-keras-layers/
#     def __init__(self, filters=None, strides=(1, 1), projection=False, **kwargs):
#         # self.filters = filters
#         self.strides = strides
#         self.projection = projection
#         if projection or strides != (1, 1):
#             self.shortcut = Conv2D(filters * 4, (1, 1), padding='same',
#                 kernel_regularizer=l2(_weight_decay), use_bias=False)
#             if not projection:
#                 self.avgpool = AvgPool2D((2, 2), strides=(2, 2), padding='same')

#         self.conv_0 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
#             kernel_regularizer=l2(_weight_decay), use_bias=False)
#         self.conv_1 = Conv2D(filters, (3, 3), strides=strides, padding='same',
#             kernel_regularizer=l2(_weight_decay), use_bias=False)
#         self.conv_2 = Conv2D(filters * 4, (1, 1), strides=(1, 1), padding='same',
#             kernel_regularizer=l2(_weight_decay), use_bias=False)
#         self.bn_0 = BatchNormalization(momentum=0.9, epsilon=1e-5)
#         self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
#         self.bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
#         self.activation0 = Activation('relu')
#         self.activation1 = Activation('relu')
#         self.activation2 = Activation('relu')
#         super(BottleneckBlock, self).__init__(**kwargs)
    
#     def __call__(self, inputs, training):
#         print(training)
#         res = self.bn_0(inputs, training=training)
#         res = self.activation0(res)
        
#         if self.projection:
#             shortcut = self.shortcut(res)
#         elif self.strides != (1, 1):
#             shortcut = self.avgpool(res) # 长宽除以2
#             shortcut = self.shortcut(shortcut)
#         else:
#             shortcut = res
        
#         res = self.conv_0(res)

#         res = self.bn_1(res, training=training)
#         res = self.activation1(res)
#         res = self.conv_1(res)

#         res = self.bn_2(res, training=training)
#         res = self.activation2(res)
#         res = self.conv_2(res)

#         output = res + shortcut
#         return res
    
    # 不写貌似也没啥问题
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[1]/self.strides[0], input_shape[2]/self.strides[1], input_shape[3])

def BottleneckBlock(inputs, filters=None, strides=(1, 1), projection=False, training=True, unit= None):
    prefix = 'bottleneck{}_{}/'.format(*unit)

    res = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix+"bn_0")(inputs, training=training)
    res = Activation('relu')(res)

    if projection:
        shortcut = Conv2D(filters * 4, (1, 1), padding='same', name=prefix+"projection",
            kernel_regularizer=l2(_weight_decay), use_bias=False)(res)
    elif strides != (1, 1):
        shortcut = AvgPool2D((2, 2), strides=(2, 2), padding='same', name=prefix+"avgpool")(res)
        shortcut = Conv2D(filters * 4, (1, 1), padding='same', name=prefix+"conv_shortcut",
            kernel_regularizer=l2(_weight_decay), use_bias=False)(shortcut)
    else:
        shortcut = res
    
    res = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=prefix+"conv_0",
            kernel_regularizer=l2(_weight_decay), use_bias=False)(res)
    
    res = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix+"bn_1")(res, training=training)
    res = Activation('relu')(res)
    res = Conv2D(filters, (3, 3), strides=strides, padding='same', name=prefix+"conv_1",
            kernel_regularizer=l2(_weight_decay), use_bias=False)(res)
        
    res = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix+"bn_2")(res, training=training)
    res = Activation('relu')(res)
    res = Conv2D(filters * 4, (1, 1), strides=(1, 1), padding='same', name=prefix+"conv_2",
            kernel_regularizer=l2(_weight_decay), use_bias=False)(res)
    
    output = keras.layers.add([res, shortcut], name=prefix+"add")
    return output


def inference(inputs, training):
    block_num = 3, 4, 6, 3
    filters_num = 64, 128, 256, 512

    res = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same',
            kernel_regularizer=l2(_weight_decay), use_bias=False)(inputs)
    res = MaxPool2D((3, 3), strides=(2, 2), padding='same')(res)

    for i in range(1, 5):
        if i == 1:
            res = BottleneckBlock(res, filters_num[i - 1], projection=True, training=training, unit=(i, 0))
        else:
            res = BottleneckBlock(res, filters_num[i - 1], strides=(2, 2), training=training, unit=(i, 0))
        for j in range(1, block_num[i - 1]):
            res = BottleneckBlock(res, filters_num[i - 1], training=training, unit=(i, j))
    
    res = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)(res, training=training)
    res = Activation('relu')(res)
    res = GlobalAvgPool2D()(res)
    res = Dense(_category_num, name='fully_connected', activation='softmax',
            kernel_regularizer=l2(_weight_decay), use_bias=False)(res)
    return res
    
if __name__=='__main__':
    img_input = Input(shape=(224,224,3))
    output = inference(img_input, training=True)
    model = keras.models.Model(img_input, output)

    print(model.summary())