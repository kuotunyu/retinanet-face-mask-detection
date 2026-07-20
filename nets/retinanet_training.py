import tensorflow as tf
from keras import backend as K

# 自訂keras的loss
def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, num_classes+1]
        #   y_pred [batch_size, num_anchor, num_classes]
        #   num_anchor就是一個位置有幾個Anchor box
        #---------------------------------------------------#
        # 下一行是用最後一個維度的最後一個值來判斷是 正樣本(前景)還是負樣本(背景)
        labels         = y_true[:, :, :-1]   # 那個-1是index ，從一開始到最後一個(不包括)
        #---------------------------------------------------#
        #   -1 是需要忽略的, 0 是背景, 1 是存在目標
        #---------------------------------------------------#
        # anchor_state的意思就是這個anchor box是ignore或前景或背景
        anchor_state   = y_true[:, :, -1]   # 最後一個維度(第3個維度)只抓最後一個位置的值
        classification = y_pred  # classification裡面放了每個bbox的預測值p
        # 等一下會拿 GT:labels 去和 預測:classification 對比來得到loss

        # 把要算loss的樣本 (可能是正或負樣本) 抓出來
        # 只要第3個維度只抓最後一個位置的值(也就是anchor_state)不是-1 就是正或負樣本(也就是要算loss的樣本)
        indices        = tf.where(K.not_equal(anchor_state, -1))  # 得到index
        labels         = tf.gather_nd(labels, indices)  # 用index取值 取要算loss的樣本的GT
        classification = tf.gather_nd(classification, indices)  # 用index取值 預測結果

        # 計算每一個先驗框應該有的權重
        alpha_factor = K.ones_like(labels) * alpha  # 論文設定: 正樣本時alpha_t 就是 alpha
        # 論文設定: 負樣本時alpha_t 就是 1-alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)

        # 下一行好像就是論文中的 (1-p_t)
        # 論文設定: 正樣本時p_t 就是 p， 負樣本時p_t 就是 1-p
        focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
        # 下一行應該就是論文中的 alpha乘上(1-p_t)^{gamma}
        focal_weight = alpha_factor * focal_weight ** gamma
        # 此時的focal_weight還差一個log(p_t)就是完整的FL了

        # 將權重乘上所求得的交叉熵，其中K.binary_crossentropy就是log(p_t)
        cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

        # 標準化，實際上是正樣本的數量
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)
        # 此時normalizer就是有幾個正樣本
        
        # 將所獲得的loss除上正樣本的數量，不是用全部樣本是因為很多負樣本都是簡單樣本不用來佔分母
        loss = K.sum(cls_loss) / normalizer
        return loss
    return _focal

def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2
    def _smooth_l1(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 4+1]
        #   y_pred [batch_size, num_anchor, 4]
        #---------------------------------------------------#
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找出存在目標的先驗框
        indices           = tf.where(K.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 計算smooth L1損失
        regression_diff = regression - regression_target
        regression_diff = K.abs(regression_diff)
        regression_loss = tf.where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 將所得的 loss 除以正樣本數量
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())
        return K.sum(regression_loss) / normalizer / 4
    return _smooth_l1
