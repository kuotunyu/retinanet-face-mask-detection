#--------------------------------------------------------#
#   印出 RetinaNet 網路結構與每一層的 index、名稱，
#   方便對照 train.py 凍結 backbone 時的 freeze_layers 設定
#--------------------------------------------------------#
import argparse

from nets.retinanet import resnet_retinanet
from utils.config import cfg_get, load_config, resolve_path
from utils.utils import get_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Print the RetinaNet model summary and layer indexes.')
    parser.add_argument('--config', default='configs/mask_retinanet.yaml',
                        help='Project config path.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    input_shape = cfg_get(config, 'model.input_shape', [600, 600])
    classes_path = resolve_path(cfg_get(config, 'paths.classes_path'))
    _, num_classes = get_classes(classes_path)

    model = resnet_retinanet([input_shape[0], input_shape[1], 3], num_classes)
    model.summary()

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
