import json
import argparse
from collections import defaultdict
import numpy as np


def box_iou_xyxy(boxes):
    if len(boxes) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    union = area[:, None] + area[None, :] - inter + 1e-6
    iou = inter / union
    np.fill_diagonal(iou, 0.0)
    return iou


def normalize_dict(d):
    values = np.array(list(d.values()), dtype=np.float32)
    min_v, max_v = values.min(), values.max()
    out = {}
    for k, v in d.items():
        out[k] = float((v - min_v) / (max_v - min_v + 1e-6))
    return out


def main(args):
    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_info = {img["id"]: img for img in coco["images"]}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        anns_by_img[ann["image_id"]].append([x, y, x + w, y + h])

    num_boxes_score = {}
    mean_max_iou_score = {}
    overlap_pair_score = {}

    for image_id in image_info.keys():
        boxes = anns_by_img.get(image_id, [])
        n = len(boxes)

        if n <= 1:
            num_boxes_score[image_id] = float(n)
            mean_max_iou_score[image_id] = 0.0
            overlap_pair_score[image_id] = 0.0
            continue

        iou = box_iou_xyxy(boxes)
        max_iou = iou.max(axis=1)

        num_boxes_score[image_id] = float(n)
        mean_max_iou_score[image_id] = float(max_iou.mean())
        overlap_pair_score[image_id] = float((iou > args.iou_thr).sum() / 2.0)

    num_boxes_score = normalize_dict(num_boxes_score)
    mean_max_iou_score = normalize_dict(mean_max_iou_score)
    overlap_pair_score = normalize_dict(overlap_pair_score)

    result = {}
    for image_id in image_info.keys():
        score = (
            args.w_num * num_boxes_score[image_id]
            + args.w_iou * mean_max_iou_score[image_id]
            + args.w_pair * overlap_pair_score[image_id]
        )

        result[str(image_id)] = {
            "score": float(score),
            "num_score": float(num_boxes_score[image_id]),
            "iou_score": float(mean_max_iou_score[image_id]),
            "pair_score": float(overlap_pair_score[image_id]),
            "num_boxes": len(anns_by_img.get(image_id, [])),
            "file_name": image_info[image_id]["file_name"],
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    scores = np.array([v["score"] for v in result.values()])
    print(f"Saved to: {args.out}")
    print(f"images: {len(result)}")
    print(f"score min/mean/max: {scores.min():.4f} / {scores.mean():.4f} / {scores.max():.4f}")
    print(f"high score >= 0.7: {(scores >= 0.7).sum()}")
    print(f"medium score 0.3~0.7: {((scores >= 0.3) & (scores < 0.7)).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--out", type=str, default="train_occlusion_score.json")
    parser.add_argument("--iou-thr", type=float, default=0.05)

    parser.add_argument("--w-num", type=float, default=0.35)
    parser.add_argument("--w-iou", type=float, default=0.45)
    parser.add_argument("--w-pair", type=float, default=0.20)

    args = parser.parse_args()
    main(args)