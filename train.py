import argparse
import os
import random

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from nets.retinanet import resnet_retinanet
from nets.retinanet_training import focal, smooth_l1
from utils.anchors import get_anchors
from utils.callbacks import LossHistory, ExponentDecayScheduler
from utils.config import cfg_get, load_config, parse_int_list, resolve_path, str2bool
from utils.dataloader import RetinanetDatasets
from utils.utils import get_classes


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaNet for face mask detection.')
    parser.add_argument('--config', default='configs/mask_retinanet.yaml',
                        help='Project config path.')
    parser.add_argument('--classes-path', default=None,
                        help='Class names txt path.')
    parser.add_argument('--pretrained-weights', default=None,
                        help='Pretrained or resume weights path. Use an empty string to train from scratch.')
    parser.add_argument('--input-shape', default=None,
                        help='Input size as "height,width", for example "600,600".')
    parser.add_argument('--anchors-size', default=None,
                        help='Anchor base sizes as comma-separated integers.')
    parser.add_argument('--train-annotations', default=None,
                        help='Training annotation txt path.')
    parser.add_argument('--val-annotations', default=None,
                        help='Validation annotation txt path.')
    parser.add_argument('--log-dir', default=None,
                        help='Directory for TensorBoard logs and checkpoints.')
    parser.add_argument('--freeze-train', type=str2bool, default=None,
                        help='Whether to run the frozen-backbone stage first.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for dataset shuffling and initialization.')
    return parser.parse_args()


def set_random_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        if hasattr(tf, 'set_random_seed'):
            tf.set_random_seed(seed)
    except Exception:
        pass


def build_settings(args):
    config = load_config(args.config)

    settings = {
        'classes_path': args.classes_path or cfg_get(config, 'paths.classes_path'),
        'model_path': args.pretrained_weights
                      if args.pretrained_weights is not None
                      else cfg_get(config, 'paths.pretrained_model_path', ''),
        'input_shape': parse_int_list(args.input_shape)
                       if args.input_shape
                       else cfg_get(config, 'model.input_shape', [600, 600]),
        'anchors_size': parse_int_list(args.anchors_size)
                        if args.anchors_size
                        else cfg_get(config, 'model.anchors_size', [32, 64, 128, 256, 512]),
        'train_annotation_path': args.train_annotations or cfg_get(config, 'paths.train_annotation_path'),
        'val_annotation_path': args.val_annotations or cfg_get(config, 'paths.val_annotation_path'),
        'log_dir': args.log_dir or cfg_get(config, 'paths.log_dir', 'logs'),
        'init_epoch': cfg_get(config, 'training.init_epoch', 0),
        'freeze_epoch': cfg_get(config, 'training.freeze_epoch', 50),
        'unfreeze_epoch': cfg_get(config, 'training.unfreeze_epoch', 100),
        'freeze_batch_size': cfg_get(config, 'training.freeze_batch_size', 16),
        'unfreeze_batch_size': cfg_get(config, 'training.unfreeze_batch_size', 4),
        'freeze_lr': cfg_get(config, 'training.freeze_lr', 1e-4),
        'unfreeze_lr': cfg_get(config, 'training.unfreeze_lr', 1e-5),
        'freeze_train': args.freeze_train
                        if args.freeze_train is not None
                        else cfg_get(config, 'training.freeze_train', True),
        'num_workers': cfg_get(config, 'training.num_workers', 1),
        'seed': args.seed if args.seed is not None else cfg_get(config, 'training.seed', None),
        'lr_decay_rate': cfg_get(config, 'training.lr_decay_rate', 0.96),
        'early_stopping_patience': cfg_get(config, 'training.early_stopping_patience', 10),
        'save_best_only': cfg_get(config, 'training.save_best_only', True),
    }

    for key in ['classes_path', 'model_path', 'train_annotation_path', 'val_annotation_path', 'log_dir']:
        settings[key] = resolve_path(settings[key])
    return settings


def run_training_stage(model, train_lines, val_lines, anchors, num_classes, settings,
                       batch_size, learning_rate, start_epoch, end_epoch, callbacks):
    num_train = len(train_lines)
    num_val = len(val_lines)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('資料集過小，無法訓練，請擴充資料集。')

    model.compile(
        loss={'regression': smooth_l1(), 'classification': focal()},
        optimizer=Adam(lr=learning_rate, clipnorm=1e-2)
    )

    train_dataloader = RetinanetDatasets(
        train_lines, settings['input_shape'], anchors, batch_size, num_classes, train=True
    )
    val_dataloader = RetinanetDatasets(
        val_lines, settings['input_shape'], anchors, batch_size, num_classes, train=False
    )

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(
        generator=train_dataloader,
        steps_per_epoch=epoch_step,
        validation_data=val_dataloader,
        validation_steps=epoch_step_val,
        epochs=end_epoch,
        initial_epoch=start_epoch,
        use_multiprocessing=settings['num_workers'] > 1,
        workers=settings['num_workers'],
        callbacks=callbacks
    )


if __name__ == "__main__":
    args = parse_args()
    settings = build_settings(args)
    set_random_seed(settings['seed'])
    os.makedirs(settings['log_dir'], exist_ok=True)

    class_names, num_classes = get_classes(settings['classes_path'])
    anchors = get_anchors(settings['input_shape'], settings['anchors_size'])

    model = resnet_retinanet((settings['input_shape'][0], settings['input_shape'][1], 3), num_classes)
    if settings['model_path']:
        print('Load weights {}.'.format(settings['model_path']))
        model.load_weights(settings['model_path'], by_name=True, skip_mismatch=True)

    logging = TensorBoard(log_dir=settings['log_dir'])
    checkpoint = ModelCheckpoint(
        os.path.join(settings['log_dir'], 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=settings['save_best_only'],
        period=1
    )
    reduce_lr = ExponentDecayScheduler(decay_rate=settings['lr_decay_rate'], verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=settings['early_stopping_patience'],
        verbose=1
    )
    loss_history = LossHistory(settings['log_dir'])
    callbacks = [logging, checkpoint, reduce_lr, early_stopping, loss_history]

    with open(settings['train_annotation_path'], encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(settings['val_annotation_path'], encoding='utf-8') as f:
        val_lines = f.readlines()

    if settings['freeze_train']:
        freeze_layers = 174
        for i in range(freeze_layers):
            model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

        run_training_stage(
            model=model,
            train_lines=train_lines,
            val_lines=val_lines,
            anchors=anchors,
            num_classes=num_classes,
            settings=settings,
            batch_size=settings['freeze_batch_size'],
            learning_rate=settings['freeze_lr'],
            start_epoch=settings['init_epoch'],
            end_epoch=settings['freeze_epoch'],
            callbacks=callbacks
        )

        for i in range(freeze_layers):
            model.layers[i].trainable = True

    run_training_stage(
        model=model,
        train_lines=train_lines,
        val_lines=val_lines,
        anchors=anchors,
        num_classes=num_classes,
        settings=settings,
        batch_size=settings['unfreeze_batch_size'],
        learning_rate=settings['unfreeze_lr'],
        start_epoch=settings['freeze_epoch'] if settings['freeze_train'] else settings['init_epoch'],
        end_epoch=settings['unfreeze_epoch'],
        callbacks=callbacks
    )
