import argparse
import os
import random
import xml.etree.ElementTree as ET

from utils.config import cfg_get, load_config, resolve_path
from utils.utils import get_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Create VOC split files and RetinaNet annotation txt files.')
    parser.add_argument('--config', default='configs/mask_retinanet.yaml',
                        help='Project config path.')
    parser.add_argument('--mode', type=int, default=None,
                        help='0 full flow, 1 ImageSets txt only, 2 train/val annotation txt only.')
    parser.add_argument('--classes-path', default=None,
                        help='Class names txt path.')
    parser.add_argument('--vocdevkit-path', default=None,
                        help='VOCdevkit root path.')
    parser.add_argument('--voc-year', default=None,
                        help='VOC dataset year folder suffix, for example 2007.')
    parser.add_argument('--trainval-percent', type=float, default=None,
                        help='Train+val ratio among all annotations.')
    parser.add_argument('--train-percent', type=float, default=None,
                        help='Train ratio within train+val annotations.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for split generation.')
    return parser.parse_args()


def build_settings(args):
    config = load_config(args.config)
    settings = {
        'annotation_mode': args.mode if args.mode is not None else cfg_get(config, 'annotation.mode', 0),
        'classes_path': args.classes_path or cfg_get(config, 'paths.classes_path'),
        'vocdevkit_path': args.vocdevkit_path or cfg_get(config, 'paths.vocdevkit_path', 'VOCdevkit'),
        'voc_year': args.voc_year or cfg_get(config, 'annotation.voc_year', '2007'),
        'trainval_percent': args.trainval_percent
                            if args.trainval_percent is not None
                            else cfg_get(config, 'annotation.trainval_percent', 0.8),
        'train_percent': args.train_percent
                         if args.train_percent is not None
                         else cfg_get(config, 'annotation.train_percent', 0.8),
        'seed': args.seed if args.seed is not None else cfg_get(config, 'training.seed', 0),
    }
    settings['classes_path'] = resolve_path(settings['classes_path'])
    settings['vocdevkit_path'] = resolve_path(settings['vocdevkit_path'])
    return settings


def convert_annotation(vocdevkit_path, voc_year, classes, image_id, list_file):
    annotation_path = os.path.join(vocdevkit_path, 'VOC%s/Annotations/%s.xml' % (voc_year, image_id))
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (
            int(float(xmlbox.find('xmin').text)),
            int(float(xmlbox.find('ymin').text)),
            int(float(xmlbox.find('xmax').text)),
            int(float(xmlbox.find('ymax').text))
        )
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def generate_imagesets(settings):
    print("Generate txt in ImageSets.")
    xmlfilepath = os.path.join(settings['vocdevkit_path'], 'VOC%s/Annotations' % settings['voc_year'])
    save_base_path = os.path.join(settings['vocdevkit_path'], 'VOC%s/ImageSets/Main' % settings['voc_year'])
    os.makedirs(save_base_path, exist_ok=True)

    total_xml = sorted([f for f in os.listdir(xmlfilepath) if f.endswith(".xml")])
    num = len(total_xml)
    indexes = list(range(num))
    trainval_count = int(num * settings['trainval_percent'])
    train_count = int(trainval_count * settings['train_percent'])
    trainval = random.sample(indexes, trainval_count)
    train = random.sample(trainval, train_count)

    print("train and val size", trainval_count)
    print("train size", train_count)

    paths = {
        'trainval': os.path.join(save_base_path, 'trainval.txt'),
        'test': os.path.join(save_base_path, 'test.txt'),
        'train': os.path.join(save_base_path, 'train.txt'),
        'val': os.path.join(save_base_path, 'val.txt'),
    }
    with open(paths['trainval'], 'w', encoding='utf-8') as ftrainval, \
         open(paths['test'], 'w', encoding='utf-8') as ftest, \
         open(paths['train'], 'w', encoding='utf-8') as ftrain, \
         open(paths['val'], 'w', encoding='utf-8') as fval:
        for i in indexes:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)
    print("Generate txt in ImageSets done.")


def generate_train_annotations(settings, classes):
    print("Generate 2007_train.txt and 2007_val.txt for train.")
    for image_set in ['train', 'val']:
        imageset_path = os.path.join(
            settings['vocdevkit_path'],
            'VOC%s/ImageSets/Main/%s.txt' % (settings['voc_year'], image_set)
        )
        with open(imageset_path, encoding='utf-8') as f:
            image_ids = f.read().strip().split()
        output_path = '%s_%s.txt' % (settings['voc_year'], image_set)
        with open(output_path, 'w', encoding='utf-8') as list_file:
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (
                    os.path.abspath(settings['vocdevkit_path']),
                    settings['voc_year'],
                    image_id
                ))
                convert_annotation(settings['vocdevkit_path'], settings['voc_year'], classes, image_id, list_file)
                list_file.write('\n')
    print("Generate 2007_train.txt and 2007_val.txt for train done.")


if __name__ == "__main__":
    args = parse_args()
    settings = build_settings(args)
    random.seed(settings['seed'])
    classes, _ = get_classes(settings['classes_path'])

    if settings['annotation_mode'] == 0 or settings['annotation_mode'] == 1:
        generate_imagesets(settings)

    if settings['annotation_mode'] == 0 or settings['annotation_mode'] == 2:
        generate_train_annotations(settings, classes)
