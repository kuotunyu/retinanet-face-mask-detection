import argparse
import os
import xml.etree.ElementTree as ET

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from PIL import Image
from tqdm import tqdm

from retinanet import Retinanet
from utils.config import cfg_get, load_config, resolve_path, str2bool
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RetinaNet mAP on a VOC-format dataset.')
    parser.add_argument('--config', default='configs/mask_retinanet.yaml',
                        help='Project config path.')
    parser.add_argument('--weights', default=None,
                        help='Inference weights path.')
    parser.add_argument('--classes-path', default=None,
                        help='Class names txt path.')
    parser.add_argument('--vocdevkit-path', default=None,
                        help='VOCdevkit root path.')
    parser.add_argument('--map-out-path', default=None,
                        help='Output directory for mAP intermediate files and results.')
    parser.add_argument('--map-mode', type=int, default=None,
                        help='0 full flow, 1 predictions, 2 ground truth, 3 VOC mAP, 4 COCO mAP.')
    parser.add_argument('--min-overlap', type=float, default=None,
                        help='IoU threshold for VOC mAP. 0.5 means mAP@0.5.')
    parser.add_argument('--map-vis', type=str2bool, default=None,
                        help='Whether to copy evaluated images into map_out/images-optional.')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Confidence threshold used when collecting detections for mAP.')
    parser.add_argument('--nms-iou', type=float, default=None,
                        help='NMS IoU threshold used when collecting detections for mAP.')
    return parser.parse_args()


def build_settings(args):
    config = load_config(args.config)
    settings = {
        'classes_path': args.classes_path or cfg_get(config, 'paths.classes_path'),
        'model_path': args.weights or cfg_get(config, 'paths.inference_model_path'),
        'vocdevkit_path': args.vocdevkit_path or cfg_get(config, 'paths.vocdevkit_path', 'VOCdevkit'),
        'map_out_path': args.map_out_path or cfg_get(config, 'paths.map_out_path', 'map_out'),
        'map_mode': args.map_mode if args.map_mode is not None else cfg_get(config, 'evaluation.map_mode', 0),
        'min_overlap': args.min_overlap if args.min_overlap is not None else cfg_get(config, 'evaluation.min_overlap', 0.5),
        'map_vis': args.map_vis if args.map_vis is not None else cfg_get(config, 'evaluation.map_vis', False),
        'confidence': args.confidence if args.confidence is not None else cfg_get(config, 'evaluation.confidence', 0.01),
        'nms_iou': args.nms_iou if args.nms_iou is not None else cfg_get(config, 'evaluation.nms_iou', 0.5),
        'voc_year': cfg_get(config, 'annotation.voc_year', '2007'),
    }
    for key in ['classes_path', 'model_path', 'vocdevkit_path', 'map_out_path']:
        settings[key] = resolve_path(settings[key])
    return settings


if __name__ == "__main__":
    args = parse_args()
    settings = build_settings(args)

    test_path = os.path.join(
        settings['vocdevkit_path'],
        'VOC{}/ImageSets/Main/test.txt'.format(settings['voc_year'])
    )
    with open(test_path, encoding='utf-8') as f:
        image_ids = f.read().strip().split()

    for sub in ['ground-truth', 'detection-results', 'images-optional']:
        os.makedirs(os.path.join(settings['map_out_path'], sub), exist_ok=True)

    class_names, _ = get_classes(settings['classes_path'])

    if settings['map_mode'] == 0 or settings['map_mode'] == 1:
        print("Load model.")
        retinanet = Retinanet(
            config_path=args.config,
            model_path=settings['model_path'],
            classes_path=settings['classes_path'],
            confidence=settings['confidence'],
            nms_iou=settings['nms_iou']
        )
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(
                settings['vocdevkit_path'],
                'VOC{}/JPEGImages/{}.jpg'.format(settings['voc_year'], image_id)
            )
            image = Image.open(image_path)
            if settings['map_vis']:
                image.save(os.path.join(settings['map_out_path'], "images-optional/" + image_id + ".jpg"))
            retinanet.get_map_txt(image_id, image, class_names, settings['map_out_path'])
        print("Get predict result done.")

    if settings['map_mode'] == 0 or settings['map_mode'] == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            output_path = os.path.join(settings['map_out_path'], "ground-truth/" + image_id + ".txt")
            with open(output_path, "w", encoding='utf-8') as new_f:
                annotation_path = os.path.join(
                    settings['vocdevkit_path'],
                    'VOC{}/Annotations/{}.xml'.format(settings['voc_year'], image_id)
                )
                root = ET.parse(annotation_path).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if settings['map_mode'] == 0 or settings['map_mode'] == 3:
        print("Get map.")
        get_map(settings['min_overlap'], True, path=settings['map_out_path'])
        print("Get map done.")

    if settings['map_mode'] == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=settings['map_out_path'])
        print("Get map done.")
