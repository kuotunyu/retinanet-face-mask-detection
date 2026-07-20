import numpy as np
import tensorflow as tf
import keras.backend as K

class BBoxUtility(object):
    def __init__(self, num_classes, nms_thresh=0.45, top_k=300):
        self.num_classes    = num_classes
        self._nms_thresh    = nms_thresh
        self._top_k         = top_k
        self.boxes          = K.placeholder(dtype='float32', shape=(None, 4))
        self.scores         = K.placeholder(dtype='float32', shape=(None,))
        self.nms            = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)
        self.sess           = K.get_session()

    def bbox_iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)

        iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
        return iou

    def efficientdet_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把 y 軸放前面，方便預測框與圖片的高寬相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   這裡求出的 offset 是圖片有效區域相對於圖片左上角的偏移量
            #   new_shape 是縮放後的高寬
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_boxes(self, mbox_loc, anchors, variance=0.2):
        # 利用座標點 獲得先驗框的寬與高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]

        # 取得真實框的左上角與右下角
        decode_bbox_xmin = mbox_loc[:,0] * anchor_width * variance + anchors[:, 0]
        decode_bbox_ymin = mbox_loc[:,1] * anchor_height * variance + anchors[:, 1]
        decode_bbox_xmax = mbox_loc[:,2] * anchor_width * variance + anchors[:, 2]
        decode_bbox_ymax = mbox_loc[:,3] * anchor_height * variance + anchors[:, 3]

        # 把左上角與右下角座標堆疊起來
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 限制在 0 與 1 之間
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, confidence=0.5):
        #---------------------------------------------------#
        #   獲得回歸預測結果
        #---------------------------------------------------#
        mbox_loc    = predictions[0]
        #---------------------------------------------------#
        #   獲得各類別的信心度
        #---------------------------------------------------#
        mbox_conf   = predictions[1]

        results     = [None for _ in range(len(mbox_loc))]
        #--------------------------------------------------------------------------------#
        #   逐張圖片處理；predict.py 一次只輸入一張圖片，所以這個迴圈只會跑一次
        #--------------------------------------------------------------------------------#
        for i in range(len(mbox_loc)):
            #--------------------------------#
            #   利用回歸結果(.decode_boxes)對先驗框進行解碼
            #--------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors)

            class_conf  = np.expand_dims(np.max(mbox_conf[i], 1), -1)     # bbox的信心度
            class_pred  = np.expand_dims(np.argmax(mbox_conf[i], 1), -1)  # bbox的預測類別
            #--------------------------------#
            #   判斷信心度是否大於門檻要求，有過門檻的bbox才留下來
            #--------------------------------#
            conf_mask       = (class_conf >= confidence)[:, 0]

            #--------------------------------#
            #   堆疊預測結果
            #--------------------------------#
            detections      = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]), 1)
            unique_labels   = np.unique(detections[:,-1])

            #-------------------------------------------------------------------#
            #   逐類別執行 NMS：非極大值抑制會在同一區域內
            #   留下同類別中得分最高的框，
            #   逐類別處理可以對每個類別分別做 NMS。
            #-------------------------------------------------------------------#
            for c in unique_labels:
                #------------------------------------------#
                #   取出該類別通過信心度篩選的全部預測結果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                #------------------------------------------#
                #   使用 TensorFlow 內建的 NMS 速度較快
                #------------------------------------------#
                idx             = self.sess.run(self.nms, feed_dict={self.boxes: detections_class[:, :4], self.scores: detections_class[:, 4]})
                max_detections  = detections_class[idx]

                results[i] = max_detections if results[i] is None else np.concatenate((results[i], max_detections), axis = 0)

            if results[i] is not None:  # 調整回原圖大小(讀入時可能補灰條)
                results[i] = np.array(results[i])
                box_xy, box_wh = (results[i][:, 0:2] + results[i][:, 2:4])/2, results[i][:, 2:4] - results[i][:, 0:2]
                results[i][:, :4] = self.efficientdet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results
