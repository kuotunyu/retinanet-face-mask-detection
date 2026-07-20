import colorsys
import os
import time

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont

from nets.retinanet import resnet_retinanet
from utils.utils_bbox import BBoxUtility
from utils.utils import get_classes, resize_image, cvtColor
from utils.anchors import get_anchors
from utils.config import cfg_get, load_config, resolve_path


def _text_size(draw, text, font):
    # Pillow 10 移除了 textsize，改用 textbbox 計算文字大小
    if hasattr(draw, 'textsize'):
        return draw.textsize(text, font)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


class Retinanet(object):
    _defaults = {
        # model_path：logs/ 下的權重檔；classes_path：model_data/ 下的 txt
        # 若出現 shape 不match，請確認訓練時的 input_shape 是否一致
        "model_path"        : 'logs/ep083-loss0.241-val_loss0.267.h5',
        "classes_path"      : 'model_data/mask_classes.txt',
        "input_shape"       : [600, 600],   # 需與訓練時一致
        "confidence"        : 0.5,          # bbox 信心門限
        "nms_iou"           : 0.3,          # NMS IOU 門限，越小越嚴格
        'anchors_size'      : [32, 64, 128, 256, 512],  # 各 feature layer anchor 基礎大小
        "letterbox_image"   : False,        # 關閉 letterbox，直接 resize 效果較佳
        "font_path"         : 'model_data/simhei.ttf',
        "config_path"       : None,
        "verbose"           : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        raise ValueError("Unrecognized attribute name '{}'".format(n))

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        config_path = kwargs.pop("config_path", self.config_path)
        if config_path:
            self._apply_config(load_config(config_path))

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.model_path = resolve_path(self.model_path)
        self.classes_path = resolve_path(self.classes_path)
        self.font_path = resolve_path(self.font_path)

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = get_anchors(self.input_shape, self.anchors_size)

        # 為各類別配置不同 HSV 顏色
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)
        self.generate()

    def _apply_config(self, config):
        self.model_path = cfg_get(config, "paths.inference_model_path", self.model_path)
        self.classes_path = cfg_get(config, "paths.classes_path", self.classes_path)
        self.font_path = cfg_get(config, "paths.font_path", self.font_path)
        self.input_shape = cfg_get(config, "model.input_shape", self.input_shape)
        self.confidence = cfg_get(config, "model.confidence", self.confidence)
        self.nms_iou = cfg_get(config, "model.nms_iou", self.nms_iou)
        self.anchors_size = cfg_get(config, "model.anchors_size", self.anchors_size)
        self.letterbox_image = cfg_get(config, "model.letterbox_image", self.letterbox_image)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.retinanet = resnet_retinanet([self.input_shape[0], self.input_shape[1], 3], self.num_classes)
        self.retinanet.load_weights(model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)  # 轉換為 RGB
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds   = self.retinanet.predict(image_data)
        results = self.bbox_util.decode_box(preds, self.anchors, image_shape,
                                            self.input_shape, self.letterbox_image, confidence=self.confidence)

        if results[0] is None:
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf  = results[0][:, 4]
        top_boxes = results[0][:, :4]

        font_size = np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')
        try:
            font = ImageFont.truetype(font=self.font_path, size=font_size)
        except OSError:
            font = ImageFont.load_default()
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box
            top    = max(0, np.floor(top).astype('int32'))
            left   = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right  = min(image.size[0], np.floor(right).astype('int32'))

            label      = '{} {:.2f}'.format(predicted_class, score)
            draw       = ImageDraw.Draw(image)
            label_size = _text_size(draw, label, font)
            if self.verbose:
                print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t, bottom - t], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image      = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds    = self.retinanet.predict(image_data)
        results  = self.bbox_util.decode_box(preds, self.anchors, image_shape,
                                             self.input_shape, self.letterbox_image, confidence=self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            preds   = self.retinanet.predict(image_data)
            results = self.bbox_util.decode_box(preds, self.anchors, image_shape,
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        image_shape = np.array(np.shape(image)[0:2])
        image      = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds    = self.retinanet.predict(image_data)
        results  = self.bbox_util.decode_box(preds, self.anchors, image_shape,
                                             self.input_shape, self.letterbox_image, confidence=self.confidence)

        result_path = os.path.join(map_out_path, "detection-results/" + image_id + ".txt")
        with open(result_path, "w", encoding="utf-8") as f:
            if results[0] is None:
                return

            top_label = results[0][:, 5]
            top_conf  = results[0][:, 4]
            top_boxes = results[0][:, :4]

            for i, c in enumerate(top_label):
                predicted_class = self.class_names[int(c)]
                box             = top_boxes[i]
                score           = str(top_conf[i])

                top, left, bottom, right = box
                if predicted_class not in class_names:
                    continue
                f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
