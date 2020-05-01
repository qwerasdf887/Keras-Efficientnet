import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import math
import numpy as np
from tensorflow.keras.regularizers import l2
tf.keras.backend.set_learning_phase(1)
conv_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='truncated_normal')
dense_init = tf.keras.initializers.VarianceScaling(scale=1.0 / 3.0, mode='fan_out', distribution='uniform')

#計算根據放寬倍數的filter數量
def round_filters(filters, width_coefficient):
    multiplier = width_coefficient
    divisor = 8
    min_depth = None
    min_depth = min_depth or divisor
    filters *= multiplier
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

#計算根據深度倍數的layer
def round_repeats(repeats, depth_coefficient):
  multiplier = depth_coefficient
  return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, survival_prob):
    '''
    根據"Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    隨機Drop一些Block
    '''
    #非training階段，直接返回
    #根據機率隨機Drop某個Block
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, survival_prob) * binary_tensor
    return tf.keras.backend.in_train_phase(output, inputs)

class act_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(act_layer, self).__init__()
    def call(self, inputs):
        return tf.nn.swish(inputs)
        #return tf.nn.relu(inputs)

class SENet(tf.keras.layers.Layer):
    def __init__(self, num_filter, input_channels, se_ratio=0.25, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.reduce_filters = max(1, int(input_channels * se_ratio))
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.reduce_filters,
                                                  kernel_size=(1,1),
                                                  strides=(1,1),
                                                  padding='same',
                                                  kernel_initializer=conv_init,
                                                  kernel_regularizer=l2(1e-5))
        self.exapnd_conv = tf.keras.layers.Conv2D(filters=num_filter,
                                                  kernel_size=(1,1),
                                                  strides=(1,1),
                                                  padding='same',
                                                  kernel_initializer=conv_init,
                                                  kernel_regularizer=l2(1e-5))
        self.act = act_layer()
        
    def call(self, inputs):
        x = self.avgpool(inputs)
        x = tf.expand_dims(input=x, axis=1)
        x = tf.expand_dims(input=x, axis=1)
        x = self.reduce_conv(x)
        x = self.act(x)
        x = self.exapnd_conv(x)
        x = tf.nn.sigmoid(x)
        return inputs * x

class MBConv(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, expand_ratio,
                 kernel_size, strides, se_ratio = 0.25, drop_ratio=0.2):
        '''
        Args:
            input_channels:input channels數量
            output_channels:output channels數量
            expand_ratio : 論文中MBConv後帶的數字
            kernel_size: kernel size (3*3) or (5*5)
            strides: Conv stride 1 or 2
            se_ratio: SENet 參數
            drop_ratio:論文中的 drop connect機率
        '''
        super(MBConv, self).__init__()
        self.strides = strides
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.survival_prob = 1 - drop_ratio
        self.drop_ratio = drop_ratio
        self.expand_ration = expand_ratio
        #根據expand ratio增加filter
        self.filters = input_channels * expand_ratio
        self.conv_0 = tf.keras.layers.Conv2D(filters=self.filters,
                                             kernel_size=(1,1),
                                             strides=(1,1),
                                             padding='same',
                                             use_bias=False,
                                             kernel_initializer=conv_init,
                                             kernel_regularizer=l2(1e-5))
        self.bn_0 = tf.keras.layers.BatchNormalization()
        #depthwise convolution
        self.depth_conv_0 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                            strides=(self.strides, self.strides),
                                                            padding='same',
                                                            use_bias=False,
                                                            kernel_initializer=conv_init,
                                                            kernel_regularizer=l2(1e-5))
        self.bn_1 = tf.keras.layers.BatchNormalization()

        #SENet
        self.SENet = SENet(self.filters, self.input_channels, se_ratio=se_ratio)

        #project convolution
        self.conv_1 = tf.keras.layers.Conv2D(filters=output_channels,
                                             kernel_size=(1,1),
                                             strides=(1,1),
                                             padding='same',
                                             use_bias=False,
                                             kernel_initializer=conv_init,
                                             kernel_regularizer=l2(1e-5))
        self.bn_2 = tf.keras.layers.BatchNormalization()
        
        self.act_1 = act_layer()
        self.act_2 = act_layer()
        
    def call(self, inputs):
        #expand dim
        if self.expand_ration != 1:
            x = self.conv_0(inputs)
            x = self.bn_0(x)
            x = self.act_1(x)
        else:
            x = inputs

        #depthwise conv
        x = self.depth_conv_0(x)
        x = self.bn_1(x)
        x = self.act_2(x)

        #SENet
        x = self.SENet(x)

        x = self.conv_1(x)
        x = self.bn_2(x)

        if self.strides == 1 and self.input_channels == self.output_channels:
            if self.drop_ratio:
                x = drop_connect(x, self.survival_prob)
            x = tf.add(x, inputs)

        return x

def creat_mbconv_block(input_tensor, input_channels, output_channels,
                       layer_repeat, expand_ratio, kernel_size, strides,
                       se_ratio = 0.25, drop_ratio=0.2):
    '''
    根據參數設定MBConv1, MBConv6 ...
    '''
    #如果layer > 1，則接下來的MBConv block
    #會用output_channels當作input_channels,strides = 1
    x = MBConv(input_channels = input_channels, output_channels = output_channels,
               expand_ratio = expand_ratio, kernel_size = kernel_size,
               strides = strides, se_ratio = se_ratio, drop_ratio=drop_ratio)(input_tensor)

    for i in range(layer_repeat - 1):
        x = MBConv(input_channels = output_channels, output_channels = output_channels,
                   expand_ratio = expand_ratio, kernel_size = kernel_size,
                   strides = 1, se_ratio = se_ratio, drop_ratio=drop_ratio)(x)

    return x


def creat_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate, num_classes=1000):
    '''
    總共分為九個部分
    1.降採樣
    2~8.不同的MBConv
    9.Conv & Pooling & FC
    '''
    img_input = tf.keras.Input(shape=(resolution, resolution, 3))
    #第一部分
    x = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                               kernel_size=(3,3), strides=(2,2), padding='same',
                               use_bias=False, kernel_initializer=conv_init,
                               kernel_regularizer=l2(1e-5))(img_input)
    x = tf.keras.layers.BatchNormalization()(x)
    
    #第二部分
    #2
    x = creat_mbconv_block(x,
                           input_channels=round_filters(32, width_coefficient),
                           output_channels=round_filters(16, width_coefficient),
                           layer_repeat=round_repeats(1, depth_coefficient),
                           expand_ratio=1, kernel_size=(3,3), strides=1,
                           drop_ratio=dropout_rate)
    #3
    x = creat_mbconv_block(x,
                           input_channels=round_filters(16, width_coefficient),
                           output_channels=round_filters(24, width_coefficient),
                           layer_repeat=round_repeats(2, depth_coefficient),
                           expand_ratio=6, kernel_size=(3,3), strides=2,
                           drop_ratio=dropout_rate)

    #4
    x = creat_mbconv_block(x,
                           input_channels=round_filters(24, width_coefficient),
                           output_channels=round_filters(40, width_coefficient),
                           layer_repeat=round_repeats(2, depth_coefficient),
                           expand_ratio=6, kernel_size=(5,5), strides=2,
                           drop_ratio=dropout_rate)

    #5
    x = creat_mbconv_block(x,
                           input_channels=round_filters(40, width_coefficient),
                           output_channels=round_filters(80, width_coefficient),
                           layer_repeat=round_repeats(3, depth_coefficient),
                           expand_ratio=6, kernel_size=(3,3), strides=2,
                           drop_ratio=dropout_rate)
    
    #6
    x = creat_mbconv_block(x,
                           input_channels=round_filters(80, width_coefficient),
                           output_channels=round_filters(112, width_coefficient),
                           layer_repeat=round_repeats(3, depth_coefficient),
                           expand_ratio=6, kernel_size=(5,5), strides=1,
                           drop_ratio=dropout_rate)

    #7
    x = creat_mbconv_block(x,
                           input_channels=round_filters(112, width_coefficient),
                           output_channels=round_filters(192, width_coefficient),
                           layer_repeat=round_repeats(4, depth_coefficient),
                           expand_ratio=6, kernel_size=(5,5), strides=2,
                           drop_ratio=dropout_rate)
    
    #8
    x = creat_mbconv_block(x,
                           input_channels=round_filters(192, width_coefficient),
                           output_channels=round_filters(320, width_coefficient),
                           layer_repeat=round_repeats(1, depth_coefficient),
                           expand_ratio=6, kernel_size=(3,3), strides=1,
                           drop_ratio=dropout_rate)

    #9
    x = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                               kernel_size=(1,1), strides=(1,1), padding='same',
                               use_bias=False, kernel_regularizer=l2(1e-5))(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = act_layer()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units=num_classes, kernel_initializer=dense_init, kernel_regularizer=l2(1e-5))(x)
    x = tf.keras.layers.Activation('softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=img_input, outputs=x)

    return model


if __name__ == '__main__':
    model = creat_efficient_net(1.0, 1.0, 224, 0.3)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy())
    model.summary()