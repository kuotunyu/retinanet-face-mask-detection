# 支援四種模式，透過 --mode 切換：
#   python predict.py --mode predict
#   python predict.py --mode video --source 0
#   python predict.py --mode fps
#   python predict.py --mode dir_predict --input img/ --output img_out/
import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from retinanet import Retinanet
from utils.config import parse_int_list, str2bool

def parse_args():
    p = argparse.ArgumentParser(description='RetinaNet 口罩偵測推論腳本')
    p.add_argument('--config',   default='configs/mask_retinanet.yaml',
                   help='專案設定檔路徑')
    p.add_argument('--weights',  default=None,
                   help='推論權重路徑')
    p.add_argument('--classes-path', default=None,
                   help='類別 txt 路徑')
    p.add_argument('--input-shape', default=None,
                   help='輸入尺寸，格式為 height,width，例如 600,600')
    p.add_argument('--confidence', type=float, default=None,
                   help='bbox 信心度門檻')
    p.add_argument('--nms-iou', type=float, default=None,
                   help='NMS IoU 門檻')
    p.add_argument('--verbose', type=str2bool, default=False,
                   help='是否印出每個偵測框')
    p.add_argument('--mode',     default='predict',
                   choices=['predict', 'video', 'fps', 'dir_predict'],
                   help='執行模式')
    p.add_argument('--image',    default='',
                   help='predict 模式：單張圖片路徑；留空則進入互動式輸入')
    p.add_argument('--output-image', default='',
                   help='predict 模式：單張圖片輸出路徑；留空則直接顯示')
    p.add_argument('--source',   default='0',
                   help='video 模式：攝影機索引（整數）或影片路徑')
    p.add_argument('--save',     default='',
                   help='video 模式：輸出影片儲存路徑，留空不儲存')
    p.add_argument('--fps',      type=float, default=25.0,
                   help='video 模式：輸出影片 FPS')
    p.add_argument('--interval', type=int,   default=100,
                   help='fps 模式：推論次數，越大越準確')
    p.add_argument('--fps-image', default='figure/demo_input.jpg',
                   help='fps 模式：測試圖片路徑')
    p.add_argument('--input',    default='img/',
                   help='dir_predict 模式：輸入資料夾路徑')
    p.add_argument('--output',   default='img_out/',
                   help='dir_predict 模式：輸出資料夾路徑')
    return p.parse_args()


def build_model(args):
    kwargs = {
        'config_path': args.config,
        'verbose': args.verbose,
    }
    if args.weights:
        kwargs['model_path'] = args.weights
    if args.classes_path:
        kwargs['classes_path'] = args.classes_path
    if args.input_shape:
        kwargs['input_shape'] = parse_int_list(args.input_shape)
    if args.confidence is not None:
        kwargs['confidence'] = args.confidence
    if args.nms_iou is not None:
        kwargs['nms_iou'] = args.nms_iou
    return Retinanet(**kwargs)


def save_or_show(image, output_path):
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)
        print('Saved to:', output_path)
    else:
        image.show()


def run_predict(model, image_path='', output_path=''):
    if image_path:
        image = Image.open(image_path)
        save_or_show(model.detect_image(image), output_path)
        return

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except Exception:
            print('Open Error! Try again!')
            continue
        save_or_show(model.detect_image(image), output_path)


def run_video(model, source, save_path, video_fps):
    source  = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(source)
    out     = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size   = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out    = cv2.VideoWriter(save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("無法讀取攝影機或影片，請確認設備已連接或影片路徑正確。")

    fps = 0.0
    while True:
        t1 = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(model.detect_image(Image.fromarray(np.uint8(frame))))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps   = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", frame)
        if out:
            out.write(frame)
        if cv2.waitKey(1) & 0xff == 27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    if out:
        print("Saved to:", save_path)
        out.release()
    cv2.destroyAllWindows()


def run_fps(model, interval, image_path):
    img       = Image.open(image_path)
    tact_time = model.get_FPS(img, interval)
    print(f'{tact_time:.4f} seconds, {1/tact_time:.1f} FPS, @batch_size 1')


def run_dir_predict(model, input_dir, output_dir):
    from tqdm import tqdm
    exts = ('.bmp', '.dib', '.png', '.jpg', '.jpeg',
            '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    os.makedirs(output_dir, exist_ok=True)
    for name in tqdm(os.listdir(input_dir)):
        if name.lower().endswith(exts):
            img     = Image.open(os.path.join(input_dir, name))
            r_image = model.detect_image(img)
            r_image.save(os.path.join(output_dir, name))


if __name__ == "__main__":
    args  = parse_args()
    model = build_model(args)

    if   args.mode == 'predict':
        run_predict(model, args.image, args.output_image)
    elif args.mode == 'video':
        run_video(model, args.source, args.save, args.fps)
    elif args.mode == 'fps':
        run_fps(model, args.interval, args.fps_image)
    elif args.mode == 'dir_predict':
        run_dir_predict(model, args.input, args.output)
