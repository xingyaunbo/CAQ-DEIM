# -*- coding: utf-8 -*-
"""
根据测试集 GT 自动划分 Easy / Medium / Hard 遮挡复杂度子集

输出：
1. test_easy.json
2. test_medium.json
3. test_hard.json
4. occlusion_group_statistics.csv

说明：
- 不需要重新训练
- 只根据 GT 标注分组，不看模型预测结果
- 适合论文中的遮挡等级分组实验
"""

import os
import json
import csv
import numpy as np
from pathlib import Path


# ====================== 路径配置 ======================
GT_JSON_PATH = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\test.json"

OUT_DIR = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups"

os.makedirs(OUT_DIR, exist_ok=True)


# ====================== 权重配置 ======================
ALPHA_NUM_GT = 0.35
BETA_MAX_IOU = 0.45
GAMMA_PAIR_NUM = 0.20

# 两个 GT 框 IoU 超过该阈值，认为存在重叠/粘连关系
OVERLAP_IOU_THRESH = 0.10


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)

    union = area1 + area2 - inter + 1e-6

    return inter / union


def minmax_norm(values):
    values = np.asarray(values, dtype=np.float32)
    v_min = values.min()
    v_max = values.max()

    if v_max - v_min < 1e-8:
        return np.zeros_like(values)

    return (values - v_min) / (v_max - v_min)


def compute_image_occlusion_stats(dataset):
    image_id_to_anns = {}

    for ann in dataset["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue

        image_id = ann["image_id"]
        image_id_to_anns.setdefault(image_id, []).append(ann)

    rows = []

    for img in dataset["images"]:
        image_id = img["id"]
        file_name = img["file_name"]

        anns = image_id_to_anns.get(image_id, [])
        boxes = [xywh_to_xyxy(ann["bbox"]) for ann in anns]

        num_gt = len(boxes)
        max_iou = 0.0
        overlap_pair_num = 0

        if num_gt >= 2:
            for i in range(num_gt):
                for j in range(i + 1, num_gt):
                    iou = box_iou(boxes[i], boxes[j])
                    max_iou = max(max_iou, iou)

                    if iou >= OVERLAP_IOU_THRESH:
                        overlap_pair_num += 1

        rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "num_gt": num_gt,
            "max_iou": max_iou,
            "overlap_pair_num": overlap_pair_num,
        })

    return rows


def assign_groups(rows):
    num_gt_list = [r["num_gt"] for r in rows]
    max_iou_list = [r["max_iou"] for r in rows]
    pair_num_list = [r["overlap_pair_num"] for r in rows]

    norm_num_gt = minmax_norm(num_gt_list)
    norm_max_iou = minmax_norm(max_iou_list)
    norm_pair_num = minmax_norm(pair_num_list)

    for i, r in enumerate(rows):
        score = (
            ALPHA_NUM_GT * float(norm_num_gt[i])
            + BETA_MAX_IOU * float(norm_max_iou[i])
            + GAMMA_PAIR_NUM * float(norm_pair_num[i])
        )

        r["norm_num_gt"] = float(norm_num_gt[i])
        r["norm_max_iou"] = float(norm_max_iou[i])
        r["norm_pair_num"] = float(norm_pair_num[i])
        r["occlusion_score"] = float(score)

    rows = sorted(rows, key=lambda x: x["occlusion_score"])

    n = len(rows)
    easy_end = n // 3
    medium_end = 2 * n // 3

    easy_rows = rows[:easy_end]
    medium_rows = rows[easy_end:medium_end]
    hard_rows = rows[medium_end:]

    for r in easy_rows:
        r["group"] = "easy"

    for r in medium_rows:
        r["group"] = "medium"

    for r in hard_rows:
        r["group"] = "hard"

    return rows, easy_rows, medium_rows, hard_rows


def save_subset_json(dataset, subset_rows, save_path):
    keep_image_ids = set([r["image_id"] for r in subset_rows])

    subset = {
        "info": dataset.get("info", {}),
        "licenses": dataset.get("licenses", []),
        "images": [img for img in dataset["images"] if img["id"] in keep_image_ids],
        "annotations": [ann for ann in dataset["annotations"] if ann["image_id"] in keep_image_ids],
        "categories": dataset["categories"],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    print(f"保存子集：{save_path}")
    print(f"  images: {len(subset['images'])}")
    print(f"  annotations: {len(subset['annotations'])}")


def save_csv(rows, save_path):
    fieldnames = [
        "image_id",
        "file_name",
        "group",
        "num_gt",
        "max_iou",
        "overlap_pair_num",
        "norm_num_gt",
        "norm_max_iou",
        "norm_pair_num",
        "occlusion_score",
    ]

    rows = sorted(rows, key=lambda x: x["occlusion_score"])

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)

    print(f"保存分组统计 CSV：{save_path}")


def print_group_summary(name, rows):
    num_gt = np.array([r["num_gt"] for r in rows], dtype=np.float32)
    max_iou = np.array([r["max_iou"] for r in rows], dtype=np.float32)
    pair_num = np.array([r["overlap_pair_num"] for r in rows], dtype=np.float32)
    score = np.array([r["occlusion_score"] for r in rows], dtype=np.float32)

    print(f"\n{name.upper()} group:")
    print(f"  images: {len(rows)}")
    print(f"  avg num_gt: {num_gt.mean():.2f}")
    print(f"  avg max_iou: {max_iou.mean():.4f}")
    print(f"  avg overlap_pair_num: {pair_num.mean():.2f}")
    print(f"  avg occlusion_score: {score.mean():.4f}")


def main():
    with open(GT_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows = compute_image_occlusion_stats(dataset)
    rows, easy_rows, medium_rows, hard_rows = assign_groups(rows)

    save_subset_json(
        dataset,
        easy_rows,
        os.path.join(OUT_DIR, "test_easy.json")
    )

    save_subset_json(
        dataset,
        medium_rows,
        os.path.join(OUT_DIR, "test_medium.json")
    )

    save_subset_json(
        dataset,
        hard_rows,
        os.path.join(OUT_DIR, "test_hard.json")
    )

    save_csv(
        rows,
        os.path.join(OUT_DIR, "occlusion_group_statistics.csv")
    )

    print_group_summary("easy", easy_rows)
    print_group_summary("medium", medium_rows)
    print_group_summary("hard", hard_rows)

    print("\n全部完成。")
    print(f"输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()