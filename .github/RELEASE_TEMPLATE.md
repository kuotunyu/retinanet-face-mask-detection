# RetinaNet Face Mask Detection Weights

## Model

- Architecture: RetinaNet + ResNet50 + FPN
- Input shape: 600x600
- Classes: without_mask, with_mask, mask_weared_incorrect
- mAP@0.5: 76.05%

## Asset

- File: `ep083-loss0.241-val_loss0.267.h5`
- Size: TODO
- SHA256: TODO

## Usage

Place the file at:

```text
logs/ep083-loss0.241-val_loss0.267.h5
```

Then run:

```bash
python predict.py --config configs/mask_retinanet.yaml --image figure/demo_input.jpg --output-image outputs/demo_result.jpg
```

Generate the final notes with:

```bash
python scripts/prepare_release.py --weights logs/ep083-loss0.241-val_loss0.267.h5 --output analysis/release_notes.md
```
