#-------------------------------------------------------------#
#   搭建ResNet50網路結構
#-------------------------------------------------------------#
from keras import layers
from keras.layers import (Activation, BatchNormalization, Conv2D, Input,
                          MaxPooling2D, ZeroPadding2D)
from keras.models import Model

# 打包identity_block，負責加深網路深度 output寬高會跟input一樣
# 參數filters要給3個深度 因為一個identity_block左線有3層
# 參數stage就是現在是第幾個stage
# 參數block 是每個stage都依序有abcd...的block
# 只要不是1x1卷積的 padding都設'same' 所以feature map高寬不會變
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # bottleneck機制來減少參數 ， 1x1 -> nxn -> 1x1
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b',use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 在identity_block只要做完卷積的樣子(x) 直接和原本input的樣子(input_tensor) 相加add就可以了
    x = layers.add([x, input_tensor]) 
    x = Activation('relu')(x)
    return x


# 打包conv_block，比上面的identity_block多了右線的1x1卷積用步長調整，output寬高可能跟input不一樣
# 參數filters要給3個深度 因為一個conv_block左線也是3層
# 參數stage就是現在是第幾個stage
# 參數block 是每個stage都依序有abcd...的block
# conv_block多了一個參數strides預設是(2, 2)，
# 也就如果不強調strides=(1, 1)的話 使用conv_block就會讓feature map的高寬都變一半
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 這層卷積是strides=strides 所以可能會影響 feature map高寬
    # bottleneck機制來減少參數 ， 1x1 -> nxn -> 1x1
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a',use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 這層卷積padding='same' 所以不影響高寬
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b',use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 這層卷積預設步長是1 且kernel size是(1, 1) 所以不影響高寬
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # kernel size是(1, 1)，strides=strides讓為了讓右線高寬跟左線一樣才可以add
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1',use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


# 唯一的參數 inputs 就是這張圖片的(h, w, c) shape
def ResNet50(inputs):
    #-----------------------------------------------------------#
    #   假設input圖片是600,600,3
    #-----------------------------------------------------------#
    img_input = inputs
    x = ZeroPadding2D((3, 3))(img_input) # input圖片進來會先在周遭補值
    # 補完值變成 606, 606, 3

    # [(606-7)/2]取整數   +   1
    # 606,606,3 -> 300,300,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',use_bias=False)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 300,300,64 -> 150,150,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 上面都還是正常的卷積操作 接下來要加入resnet結構了
    # 150,150,64 -> 150,150,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # y0 = x # 就是 C2(後面不會用到 只是告訴我這邊到C2了)

    # 150,150,256 -> 75,75,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    y1 = x # 就是 C3

    # 75,75,512 -> 38,38,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    y2 = x # 就是 C4
    
    # 38,38,1024 -> 19,19,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    y3 = x # 就是 C5
    return y1, y2, y3

if __name__ == "__main__":
    inputs = Input(shape=(600, 600, 3))
    model = ResNet50(inputs)
    model.summary()
