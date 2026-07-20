#--------------------------------------------------------------------#
#   視覺化 FPN 某一層的 anchor 分布與 decode 行為，
#   用來產生 README 的 figure/anchor_distribution.png。
#   左圖：該層的網格中心，以及中央網格點的 9 個先驗框
#   右圖：先驗框套用隨機回歸偏移量 decode 之後的結果
#--------------------------------------------------------------------#
import argparse
import os

import matplotlib
import numpy as np

from utils.anchors import AnchorBox, get_img_output_length
from utils.config import cfg_get, load_config, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FPN anchor distribution for one pyramid level.')
    parser.add_argument('--config', default='configs/mask_retinanet.yaml',
                        help='Project config path.')
    parser.add_argument('--level', type=int, default=4,
                        help='FPN level index: 0=P3, 1=P4, 2=P5, 3=P6, 4=P7.')
    parser.add_argument('--output', default='figure/anchor_distribution.png',
                        help='Output image path.')
    parser.add_argument('--show', type=str2bool, default=False,
                        help='Show the figure in a window instead of saving it.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for the demo regression offsets.')
    return parser.parse_args()


def decode_boxes(mbox_loc, anchors, variance=0.2):
    #------------------------------------------------------------#
    #   與 utils/utils_bbox.py 相同的解碼方式：
    #   回歸預測值是四個角點相對於先驗框寬高的偏移量
    #------------------------------------------------------------#
    anchor_width  = anchors[:, 2] - anchors[:, 0]
    anchor_height = anchors[:, 3] - anchors[:, 1]

    decode_bbox_xmin = mbox_loc[:, 0] * anchor_width * variance + anchors[:, 0]
    decode_bbox_ymin = mbox_loc[:, 1] * anchor_height * variance + anchors[:, 1]
    decode_bbox_xmax = mbox_loc[:, 2] * anchor_width * variance + anchors[:, 2]
    decode_bbox_ymax = mbox_loc[:, 3] * anchor_height * variance + anchors[:, 3]

    return np.stack([decode_bbox_xmin, decode_bbox_ymin, decode_bbox_xmax, decode_bbox_ymax], axis=-1)


def draw_boxes(ax, boxes, indexes):
    for i in indexes:
        width  = boxes[i, 2] - boxes[i, 0]
        height = boxes[i, 3] - boxes[i, 1]
        ax.add_patch(matplotlib.patches.Rectangle(
            (boxes[i, 0], boxes[i, 1]), width, height, color='r', fill=False))


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    config       = load_config(args.config)
    input_shape  = cfg_get(config, 'model.input_shape', [600, 600])
    anchors_size = cfg_get(config, 'model.anchors_size', [32, 64, 128, 256, 512])
    strides      = [8, 16, 32, 64, 128]

    if not 0 <= args.level < len(anchors_size):
        raise ValueError('level 必須介於 0 與 {} 之間。'.format(len(anchors_size) - 1))
    level = args.level

    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    anchor_box      = AnchorBox(ratios=[0.5, 1, 2], scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    base_anchors    = anchor_box.generate_anchors(anchors_size[level])
    shifted_anchors = anchor_box.shift([feature_heights[level], feature_widths[level]], strides[level], base_anchors)

    #------------------------------------------------------------#
    #   只畫中央網格點的 9 個先驗框，避免整張圖被框蓋滿
    #------------------------------------------------------------#
    grid_h, grid_w = int(feature_heights[level]), int(feature_widths[level])
    center_cell    = (grid_h // 2) * grid_w + grid_w // 2
    sample_idx     = list(range(center_cell * anchor_box.num_anchors,
                                (center_cell + 1) * anchor_box.num_anchors))

    cell_x = (shifted_anchors[:, 0] + shifted_anchors[:, 2]) / 2
    cell_y = (shifted_anchors[:, 1] + shifted_anchors[:, 3]) / 2
    margin = anchors_size[level]

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('P{} anchors (input {}x{}, base size {})'.format(
        level + 3, input_shape[0], input_shape[1], anchors_size[level]))

    #------------------------------------------------------------#
    #   左圖：網格中心與中央網格點的 9 個先驗框
    #------------------------------------------------------------#
    ax = fig.add_subplot(121)
    ax.set_xlim(-margin, input_shape[1] + margin)
    ax.set_ylim(-margin, input_shape[0] + margin)
    ax.scatter(cell_x, cell_y, s=8)
    draw_boxes(ax, shifted_anchors, sample_idx)
    ax.invert_yaxis()
    ax.set_title('anchors')

    #------------------------------------------------------------#
    #   右圖：套用隨機回歸偏移量 decode 後的框
    #------------------------------------------------------------#
    np.random.seed(args.seed)
    random_loc   = np.random.uniform(0, 1, [len(shifted_anchors), 4]) / 10
    after_decode = decode_boxes(random_loc, shifted_anchors)

    decoded_center_x = (after_decode[:, 0] + after_decode[:, 2]) / 2
    decoded_center_y = (after_decode[:, 1] + after_decode[:, 3]) / 2

    ax = fig.add_subplot(122)
    ax.set_xlim(-margin, input_shape[1] + margin)
    ax.set_ylim(-margin, input_shape[0] + margin)
    ax.scatter(cell_x, cell_y, s=8)
    ax.scatter(decoded_center_x[sample_idx], decoded_center_y[sample_idx], s=8)
    draw_boxes(ax, after_decode, sample_idx)
    ax.invert_yaxis()
    ax.set_title('after decode')

    if args.show:
        plt.show()
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print('Saved to:', args.output)


if __name__ == "__main__":
    main()
