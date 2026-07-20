import argparse
import hashlib
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import cfg_get, load_config, resolve_path
from utils.utils import get_classes


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def build_release_notes(args):
    config = load_config(args.config)
    weights_path = resolve_path(args.weights or cfg_get(config, "paths.inference_model_path"))
    classes_path = resolve_path(args.classes_path or cfg_get(config, "paths.classes_path"))
    input_shape = cfg_get(config, "model.input_shape", [600, 600])
    class_names, _ = get_classes(classes_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError("Weights file not found: {}".format(weights_path))

    notes = []
    notes.append("# RetinaNet Face Mask Detection Weights")
    notes.append("")
    notes.append("## Model")
    notes.append("")
    notes.append("- Architecture: RetinaNet + ResNet50 + FPN")
    notes.append("- Input shape: {}x{}".format(input_shape[0], input_shape[1]))
    notes.append("- Classes: {}".format(", ".join(class_names)))
    notes.append("- mAP@0.5: 76.05%")
    notes.append("")
    notes.append("## Asset")
    notes.append("")
    notes.append("- File: `{}`".format(os.path.basename(weights_path)))
    notes.append("- Size: {:.2f} MB".format(file_size_mb(weights_path)))
    notes.append("- SHA256: `{}`".format(sha256_file(weights_path)))
    notes.append("")
    notes.append("## Usage")
    notes.append("")
    notes.append("Place the file at:")
    notes.append("")
    notes.append("```text")
    notes.append(cfg_get(config, "paths.inference_model_path"))
    notes.append("```")
    notes.append("")
    notes.append("Then run:")
    notes.append("")
    notes.append("```bash")
    notes.append("python predict.py --config configs/mask_retinanet.yaml --image figure/demo_input.jpg --output-image outputs/demo_result.jpg")
    notes.append("```")
    notes.append("")
    return "\n".join(notes)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GitHub Release notes for the trained weights.")
    parser.add_argument("--config", default="configs/mask_retinanet.yaml")
    parser.add_argument("--weights", default="", help="Weights path. Defaults to paths.inference_model_path.")
    parser.add_argument("--classes-path", default="", help="Class names path. Defaults to paths.classes_path.")
    parser.add_argument("--output", default="", help="Optional markdown output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    notes = build_release_notes(args)
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(notes)
        print("Wrote release notes:", args.output)
    else:
        print(notes)


if __name__ == "__main__":
    main()
