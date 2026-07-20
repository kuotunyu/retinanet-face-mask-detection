# GitHub Release Checklist

Use GitHub Releases for trained `.h5` weights instead of committing them to the repository.

## 1. Verify The Weight Path

Expected default:

```text
logs/ep083-loss0.241-val_loss0.267.h5
```

## 2. Generate Release Notes

```bash
python scripts/prepare_release.py \
  --weights logs/ep083-loss0.241-val_loss0.267.h5 \
  --output analysis/release_notes.md
```

The generated notes include:

- Architecture
- Input shape
- Class names
- mAP@0.5
- File size
- SHA256
- Usage command

## 3. Create GitHub Release

Suggested tag:

```text
v1.0-mask-retinanet
```

Attach:

```text
ep083-loss0.241-val_loss0.267.h5
```

Paste the generated `analysis/release_notes.md` content into the release description.

## 4. Refresh Demo Screenshots

After launching the redesigned Gradio UI:

```bash
python demo.py --config configs/mask_retinanet.yaml
```

Replace:

```text
figure/gradio_init.png
figure/gradio_result.png
```

These screenshots are part of the first impression on GitHub.
