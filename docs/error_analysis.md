# Error Analysis

Run mAP evaluation first so `map_out/ground-truth` and `map_out/detection-results` exist:

```bash
python get_map.py --config configs/mask_retinanet.yaml
```

## Threshold Sweep

Use the low-confidence detection outputs from mAP evaluation to inspect precision, recall, and F1 under different confidence thresholds:

```bash
python scripts/analyze_detections.py threshold-sweep \
  --map-out map_out \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --output analysis/threshold_sweep.csv
```

The CSV contains per-class and overall metrics:

```text
threshold,class_name,tp,fp,fn,precision,recall,f1
```

## False Positives / False Negatives

Create a CSV of missed detections and false alarms:

```bash
python scripts/analyze_detections.py errors \
  --map-out map_out \
  --threshold 0.5 \
  --output analysis/error_cases.csv
```

Optionally render annotated error images:

```bash
python scripts/analyze_detections.py errors \
  --map-out map_out \
  --threshold 0.5 \
  --images-dir VOCdevkit/VOC2007/JPEGImages \
  --render-dir analysis/error_images
```

Rendered colors:

- Green: true positive detection
- Red: false positive detection
- Amber: ground-truth box involved in false negative or class confusion

## What To Look For

- `mask_weared_incorrect` false negatives: often caused by very small or partially occluded masks.
- `without_mask` false positives: check whether uncovered lower faces or side profiles are over-triggering.
- Class confusion between `with_mask` and `mask_weared_incorrect`: usually indicates ambiguous annotation boundaries.
- Low-light or crowded images: useful examples to document limitations in the README or interview discussion.
