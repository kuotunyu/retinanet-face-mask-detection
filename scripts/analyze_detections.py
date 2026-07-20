import argparse
import csv
import os
from collections import defaultdict


def parse_box(values):
    return tuple(float(v) for v in values)


def iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    inter = width * height
    if inter == 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_ground_truth(path, include_difficult=False):
    records = defaultdict(list)
    gt_dir = os.path.join(path, "ground-truth")
    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith(".txt"):
            continue
        image_id = os.path.splitext(name)[0]
        with open(os.path.join(gt_dir, name), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                difficult = len(parts) > 5 and parts[5] == "difficult"
                if difficult and not include_difficult:
                    continue
                records[image_id].append({
                    "class_name": parts[0],
                    "box": parse_box(parts[1:5]),
                    "difficult": difficult,
                })
    return records


def read_detections(path):
    records = defaultdict(list)
    det_dir = os.path.join(path, "detection-results")
    for name in sorted(os.listdir(det_dir)):
        if not name.endswith(".txt"):
            continue
        image_id = os.path.splitext(name)[0]
        with open(os.path.join(det_dir, name), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                records[image_id].append({
                    "class_name": parts[0],
                    "score": float(parts[1]),
                    "box": parse_box(parts[2:6]),
                })
    return records


def load_map_out(path, include_difficult=False):
    return read_ground_truth(path, include_difficult), read_detections(path)


def match_image(ground_truth, detections, threshold, min_iou):
    matches = []
    unmatched_gt = set(range(len(ground_truth)))
    sorted_detections = sorted(
        [det for det in detections if det["score"] >= threshold],
        key=lambda item: item["score"],
        reverse=True
    )

    for det in sorted_detections:
        best_idx = None
        best_iou = 0.0
        for idx in unmatched_gt:
            gt = ground_truth[idx]
            if gt["class_name"] != det["class_name"]:
                continue
            current_iou = iou(det["box"], gt["box"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_idx is not None and best_iou >= min_iou:
            unmatched_gt.remove(best_idx)
            matches.append(("tp", det, ground_truth[best_idx], best_iou))
        else:
            matches.append(("fp", det, None, best_iou))

    for idx in sorted(unmatched_gt):
        matches.append(("fn", None, ground_truth[idx], 0.0))
    return matches


def summarize_threshold(ground_truth_by_image, detections_by_image, threshold, min_iou):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    image_ids = sorted(set(ground_truth_by_image) | set(detections_by_image))

    for image_id in image_ids:
        matches = match_image(
            ground_truth_by_image.get(image_id, []),
            detections_by_image.get(image_id, []),
            threshold,
            min_iou
        )
        for kind, det, gt, _ in matches:
            class_name = det["class_name"] if det else gt["class_name"]
            stats[class_name][kind] += 1
            stats["overall"][kind] += 1

    rows = []
    for class_name in sorted(stats.keys()):
        counts = stats[class_name]
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append({
            "threshold": threshold,
            "class_name": class_name,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_csv(path, rows, fieldnames):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def threshold_sweep(args):
    ground_truth, detections = load_map_out(args.map_out, args.include_difficult)
    rows = []
    for threshold in args.thresholds:
        rows.extend(summarize_threshold(ground_truth, detections, threshold, args.min_iou))

    write_csv(
        args.output,
        rows,
        ["threshold", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"]
    )
    print("Wrote threshold sweep:", args.output)


def best_other_class_iou(det, ground_truth):
    best = (None, 0.0)
    for gt in ground_truth:
        current_iou = iou(det["box"], gt["box"])
        if current_iou > best[1]:
            best = (gt, current_iou)
    return best


def error_analysis(args):
    ground_truth, detections = load_map_out(args.map_out, args.include_difficult)
    image_ids = sorted(set(ground_truth) | set(detections))
    rows = []
    render_items = []

    for image_id in image_ids:
        gt_items = ground_truth.get(image_id, [])
        det_items = detections.get(image_id, [])
        matches = match_image(gt_items, det_items, args.threshold, args.min_iou)
        for kind, det, gt, match_iou in matches:
            if kind == "tp":
                if args.include_tp:
                    rows.append(make_row(image_id, kind, det, gt, match_iou, "matched"))
                    render_items.append((image_id, kind, det, gt, match_iou))
                continue
            if kind == "fp":
                other_gt, other_iou = best_other_class_iou(det, gt_items)
                reason = "background"
                if other_gt is not None and other_iou >= args.min_iou:
                    reason = "class_confusion:{}".format(other_gt["class_name"])
                rows.append(make_row(image_id, kind, det, other_gt, other_iou, reason))
                render_items.append((image_id, kind, det, other_gt, other_iou))
            elif kind == "fn":
                rows.append(make_row(image_id, kind, None, gt, match_iou, "missed"))
                render_items.append((image_id, kind, None, gt, match_iou))

    rows = sorted(rows, key=lambda row: (row["type"], row["image_id"], row["class_name"], -float(row["score"] or 0)))
    if args.max_rows:
        rows = rows[:args.max_rows]

    write_csv(
        args.output,
        rows,
        ["image_id", "type", "reason", "class_name", "score", "iou", "box", "matched_class", "matched_box"]
    )
    print("Wrote error analysis:", args.output)

    if args.images_dir and args.render_dir:
        render_error_images(args, render_items)


def make_row(image_id, kind, det, gt, match_iou, reason):
    class_name = det["class_name"] if det else gt["class_name"]
    box = det["box"] if det else gt["box"]
    return {
        "image_id": image_id,
        "type": kind,
        "reason": reason,
        "class_name": class_name,
        "score": "{:.6f}".format(det["score"]) if det else "",
        "iou": "{:.4f}".format(match_iou),
        "box": format_box(box),
        "matched_class": gt["class_name"] if gt else "",
        "matched_box": format_box(gt["box"]) if gt else "",
    }


def format_box(box):
    return ",".join(str(int(round(value))) for value in box)


def render_error_images(args, render_items):
    from PIL import Image, ImageDraw

    os.makedirs(args.render_dir, exist_ok=True)
    count = 0
    for image_id, kind, det, gt, match_iou in render_items:
        if count >= args.max_images:
            break
        image_path = find_image(args.images_dir, image_id)
        if not image_path:
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        if gt:
            draw_box(draw, gt["box"], "#d09522", "GT {}".format(gt["class_name"]))
        if det:
            color = "#12806a" if kind == "tp" else "#c84c48"
            draw_box(draw, det["box"], color, "{} {:.2f}".format(det["class_name"], det["score"]))
        output_name = "{}_{}_iou{:.2f}.jpg".format(image_id, kind, match_iou)
        image.save(os.path.join(args.render_dir, output_name), quality=92)
        count += 1
    print("Rendered {} images to {}".format(count, args.render_dir))


def find_image(images_dir, image_id):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        path = os.path.join(images_dir, image_id + ext)
        if os.path.exists(path):
            return path
    return None


def draw_box(draw, box, color, label):
    left, top, right, bottom = [int(round(value)) for value in box]
    for offset in range(3):
        draw.rectangle([left + offset, top + offset, right - offset, bottom - offset], outline=color)
    text_y = max(0, top - 14)
    draw.rectangle([left, text_y, left + max(72, len(label) * 7), text_y + 14], fill=color)
    draw.text((left + 3, text_y + 1), label, fill="white")


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze RetinaNet VOC-format detection outputs.")
    subparsers = parser.add_subparsers(dest="command")

    sweep = subparsers.add_parser("threshold-sweep", help="Compute precision/recall/F1 across confidence thresholds.")
    sweep.add_argument("--map-out", default="map_out", help="Path containing detection-results and ground-truth folders.")
    sweep.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"))
    sweep.add_argument("--min-iou", type=float, default=0.5)
    sweep.add_argument("--include-difficult", action="store_true")
    sweep.add_argument("--output", default="analysis/threshold_sweep.csv")
    sweep.set_defaults(func=threshold_sweep)

    errors = subparsers.add_parser("errors", help="List false positives and false negatives.")
    errors.add_argument("--map-out", default="map_out", help="Path containing detection-results and ground-truth folders.")
    errors.add_argument("--threshold", type=float, default=0.5)
    errors.add_argument("--min-iou", type=float, default=0.5)
    errors.add_argument("--include-difficult", action="store_true")
    errors.add_argument("--include-tp", action="store_true", help="Also include true positives in the CSV/rendered images.")
    errors.add_argument("--max-rows", type=int, default=0, help="Limit CSV rows. 0 means no limit.")
    errors.add_argument("--images-dir", default="", help="Optional original image directory for rendering cases.")
    errors.add_argument("--render-dir", default="analysis/error_images", help="Directory for rendered error images.")
    errors.add_argument("--max-images", type=int, default=40)
    errors.add_argument("--output", default="analysis/error_cases.csv")
    errors.set_defaults(func=error_analysis)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
