import math

import keras
import keras.layers
import numpy as np
import tensorflow as tf

from nets.resnet import ResNet50


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.image.resize_images(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class PriorProbability(keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {'probability': self.probability}

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)
        return result

#-----------------------------------------#
#   Retinahead 獲得回歸預測結果
#   所有特徵層共用一個Retinahead
#-----------------------------------------#
# 這個make_last_layer_loc就是 FPN後面接的分類head，P3 P4 P5是共用這個迴歸head
def make_last_layer_loc(num_anchors, pyramid_feature_size = 256):
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size)) 
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }
    outputs = inputs
    #-----------------------------------------#
    #   論文設定要進行四次卷積，通道數均為256
    #   做4次一樣的是所以直接用迴圈
    #-----------------------------------------#
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=256,activation='relu',name='pyramid_regression_{}'.format(i),**options)(outputs)
    #-----------------------------------------#
    #   獲得回歸預測結果，並進行reshape
    #-----------------------------------------#
    # K:前景類別個數， A:Anchor box數
    outputs     = keras.layers.Conv2D(4 * num_anchors,  # 深度4xA
                                      name='pyramid_regression', **options)(outputs)
    # Reshape完 -1那個維度就是有幾個bbox，4就是xywh偏移量 => (A, 4)
    regression  = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)
    #-----------------------------------------#
    #   構建成一個模型
    #-----------------------------------------#
    regression_model = keras.models.Model(inputs=inputs, outputs=regression, name="regression_submodel")
    return regression_model

#-----------------------------------------#
#   Retinahead 獲得分類預測結果
#   所有特徵層共用一個Retinahead
#-----------------------------------------#
# 這個make_last_layer_cls就是 FPN後面接的分類head，P3 P4 P5是共用這個分類head
def make_last_layer_cls(num_classes, num_anchors, pyramid_feature_size=256):
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }
    outputs = inputs
    #-----------------------------------------#
    #   進行四次卷積，通道數均為256
    #-----------------------------------------#
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=256, activation='relu', name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None), bias_initializer='zeros', **options)(outputs)
    #-----------------------------------------#
    #   獲得分類預測結果，並進行reshape
    #-----------------------------------------#
    outputs = keras.layers.Conv2D(filters= num_classes * num_anchors,  # 深度 KxA
        kernel_initializer  = keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer    = PriorProbability(probability=0.01),
        name='pyramid_classification'.format(),
        **options
    )(outputs)
    # Reshape完 -1那個維度就是有幾個bbox，num_classes就是有幾個類別 => (A, K)
    outputs         = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    #-----------------------------------------#
    #   為了轉換成機率，使用sigmoid激活函數 做A次是非題 如果每個類別分數都很低就判斷是背景
    #-----------------------------------------#
    classification  = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    #-----------------------------------------#
    #   構建成一個模型
    #-----------------------------------------#
    classification_model = keras.models.Model(inputs=inputs, outputs=classification, name="classification_submodel")
    return classification_model

# 最好搭配示意圖
def resnet_retinanet(input_shape, num_classes, num_anchors = 9, name='retinanet'):
    inputs = keras.layers.Input(shape=input_shape)
    #-----------------------------------------#
    #   取出三個有效特徵層，分別是C3、C4、C5
    #   C3     75,75,512
    #   C4     38,38,1024
    #   C5     19,19,2048
    #-----------------------------------------#
    C3, C4, C5 = ResNet50(inputs)
    # ResNet50 是調用同的路徑下 resnet.py的def ResNet50 會return C3 C4 C5

    # 75,75,512 -> 75,75,256 ，C3先經過1x1卷積來轉換深度 方便等一下做add
    P3              = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    # 38,38,1024 -> 38,38,256 ， C4也調整深度2. 都統一到256
    P4              = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    # 19,19,2048 -> 19,19,256 ， C5也調整深度
    P5              = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    #P5 比較單純就是C5調整深度而已

    # 19,19,256 -> 38,38,256 ， 把P5做上採樣 讓hw跟C4一樣
    P5_upsampled    = UpsampleLike(name='P5_upsampled')([P5, P4])
    # 38,38,256 + 38,38,256 -> 38,38,256 ，把做完上採樣的P5和調整深度的C4做add = P4
    P4              = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    # 38,38,256 -> 75,75,256 把P4做上採樣 讓hw跟C3一樣
    P4_upsampled    = UpsampleLike(name='P4_upsampled')([P4, P3])
    # 75,75,256 + 75,75,256 -> 75,75,256 ，把做完上採樣的P4和調整深度的C3做add = P3
    P3              = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])

    # 75,75,256 -> 75,75,256
    P3              = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    # 38,38,256 -> 38,38,256
    P4              = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    # 19,19,256 -> 19,19,256
    P5              = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # 19,19,2048 -> 10,10,256 ， 原本的C5做一個 strides=2的卷積 就是P6
    P6              = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6')(C5)
    P7              = keras.layers.Activation('relu', name='C6_relu')(P6)
    # 10,10,256 -> 5,5,256
    P7              = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7')(P7)
    # P7 就是把P6再做一個 strides=2的卷積 就是P7

    features        =  [P3, P4, P5, P6, P7]

    regression_model        = make_last_layer_loc(num_anchors)
    classification_model    = make_last_layer_cls(num_classes, num_anchors)

    regressions     = []
    classifications = []
    
    #----------------------------------------------------------#
    #   將獲取到的P3, P4, P5, P6, P7傳入到
    #   Retinahead裡面進行預測，獲得回歸預測結果和分類預測結果
    #   將所有特徵層的預測結果進行堆疊
    #   P3 P4 P5那些都是用同一個regression head 同一個classification head
    #----------------------------------------------------------#
    for feature in features:  # features是[P3, P4, P5, P6, P7]
        regression      = regression_model(feature)
        # regression : 迴歸head的預測結果
        classification  = classification_model(feature)
        # classification: 分類head的預測結果

        # 把FPN不同尺度的P3 P4 P5那些的預測結果用.append收集起來 等一下轉成keras的格式
        regressions.append(regression)
        classifications.append(classification)

    #利用keras.layers.Concatenate把P3, P4, P5, P6, P7的預測結果堆疊在一起
    regressions     = keras.layers.Concatenate(axis=1, name="regression")(regressions)
    classifications = keras.layers.Concatenate(axis=1, name="classification")(classifications)

    model = keras.models.Model(inputs, [regressions, classifications], name=name)

    return model
