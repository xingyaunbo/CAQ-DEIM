"""
论文标准测试脚本 + 一猪多框统计
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import sys
import os
import json
import csv
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


# ====================== 论文测试固定配置 ======================
CFG_PATH = r"C:\Users\26930\Desktop\DEIM-main\configs\deim_dfine\deim_hgnetv2_n_coco.yml"
WEIGHT_PATH = r"C:\Users\26930\Desktop\DEIM-main\deim_outputs\deim_hgnetv2_n_coco\best_stg2.pth"
TEST_IMG_DIR = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\test"
GT_JSON_PATH = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\test.json"
PRED_JSON_PATH = r"C:\Users\26930\Desktop\DEIM-main\predict_results\coco_predictions.json"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# COCO评估建议保留很低阈值
CONF_THRESH = 0.001

# ====================== 一猪多框统计参数 ======================
# 统计一猪多框时，不建议用0.001，否则大量低置信度框会干扰统计
DUP_CONF_THRESH = 0.3

# IoU阈值：IoU >= 0.5 认为该预测框属于这个GT猪
DUP_IOU_THR = 0.5

DUP_CSV_PATH = r"C:\Users\26930\Desktop\DEIM-main\predict_results\duplicate_pig_analysis.csv"
# ==============================================================


def load_model():
    cfg = YAMLConfig(CFG_PATH, resume=WEIGHT_PATH)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')

    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print("✅ 使用EMA权重")
    else:
        state = checkpoint['model']
        print("✅ 使用普通model权重")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(DEVICE)
    model.eval()

    print("✅ 模型加载完成")
    return model


def get_image_id(filename, coco_gt):
    for img_info in coco_gt.dataset['images']:
        if img_info['file_name'] == filename:
            return img_info['id']
    return -1


def bbox_iou_xywh(box1, box2):
    """
    计算两个xywh格式框的IoU。

    box格式:
    [x, y, w, h]
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1 = x1, y1
    xa2, ya2 = x1 + w1, y1 + h1

    xb1, yb1 = x2, y2
    xb2, yb2 = x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def analyze_duplicate_pig_boxes(
    coco_gt,
    pred_results,
    conf_thr=0.30,
    iou_thr=0.50,
    save_csv_path=None
):
    """
    统计一猪多框问题。

    对每个GT猪，统计有多少个预测框与它 IoU >= iou_thr。
    如果一个GT对应多个预测框，则认为是一猪多框。

    输出：
    1. GT总数
    2. 漏检GT数量
    3. 正常检测GT数量
    4. 一猪多框GT数量
    5. 多余重复框数量
    6. 一猪多框比例
    7. 详细CSV
    """

    preds_by_img = defaultdict(list)

    for pred in pred_results:
        if pred["score"] >= conf_thr:
            preds_by_img[pred["image_id"]].append(pred)

    detail_rows = []

    total_gt = 0
    missed_gt = 0
    normal_gt = 0
    duplicate_gt = 0

    total_duplicate_boxes = 0
    total_matched_preds = 0

    duplicate_images = set()

    img_ids = coco_gt.getImgIds()

    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(ann_ids)

        gt_anns = [
            ann for ann in gt_anns
            if ann.get("iscrowd", 0) == 0
            and ann["bbox"][2] > 0
            and ann["bbox"][3] > 0
        ]

        preds = preds_by_img.get(img_id, [])

        total_gt += len(gt_anns)

        for gt_idx, gt in enumerate(gt_anns):
            gt_box = gt["bbox"]
            gt_cat = gt["category_id"]

            matched_preds = []

            for pred in preds:
                pred_box = pred["bbox"]
                pred_cat = pred["category_id"]

                # 单类别检测一般类别一致即可
                if pred_cat != gt_cat:
                    continue

                iou = bbox_iou_xywh(pred_box, gt_box)

                if iou >= iou_thr:
                    matched_preds.append({
                        "score": float(pred["score"]),
                        "iou": float(iou),
                        "bbox": pred_box
                    })

            matched_preds = sorted(
                matched_preds,
                key=lambda x: x["score"],
                reverse=True
            )

            matched_num = len(matched_preds)
            total_matched_preds += matched_num

            if matched_num == 0:
                missed_gt += 1
                status = "missed"
                duplicate_num = 0

            elif matched_num == 1:
                normal_gt += 1
                status = "normal"
                duplicate_num = 0

            else:
                duplicate_gt += 1
                duplicate_images.add(img_id)
                status = "duplicate"
                duplicate_num = matched_num - 1
                total_duplicate_boxes += duplicate_num

            detail_rows.append({
                "image_id": img_id,
                "gt_index": gt_idx,
                "category_id": gt_cat,
                "gt_bbox": gt_box,
                "matched_pred_num": matched_num,
                "duplicate_pred_num": duplicate_num,
                "status": status,
                "matched_scores": [round(x["score"], 4) for x in matched_preds],
                "matched_ious": [round(x["iou"], 4) for x in matched_preds],
                "matched_pred_bboxes": [x["bbox"] for x in matched_preds],
            })

    detected_gt = normal_gt + duplicate_gt

    print("\n================ 一猪多框统计结果 ================")
    print(f"置信度阈值 conf_thr: {conf_thr}")
    print(f"IoU匹配阈值 iou_thr: {iou_thr}")

    print(f"\nGT猪总数: {total_gt}")
    print(f"漏检GT数量: {missed_gt}")
    print(f"正常检测GT数量: {normal_gt}")
    print(f"一猪多框GT数量: {duplicate_gt}")

    print(f"\n检测到的GT数量: {detected_gt}")
    print(f"所有匹配到GT的预测框数量: {total_matched_preds}")
    print(f"多余重复框数量: {total_duplicate_boxes}")

    if total_gt > 0:
        print(f"\n漏检率 missed_gt / total_gt: {missed_gt / total_gt:.4f}")
        print(f"一猪多框GT占总GT比例 duplicate_gt / total_gt: {duplicate_gt / total_gt:.4f}")

    if detected_gt > 0:
        print(f"一猪多框GT占已检测GT比例 duplicate_gt / detected_gt: {duplicate_gt / detected_gt:.4f}")

    if total_matched_preds > 0:
        print(f"重复框占匹配预测框比例 duplicate_boxes / matched_preds: {total_duplicate_boxes / total_matched_preds:.4f}")

    print(f"\n存在一猪多框的图片数量: {len(duplicate_images)}")

    if duplicate_gt > 0:
        avg_dup_per_dup_gt = total_duplicate_boxes / duplicate_gt
        print(f"每个一猪多框GT平均多余框数量: {avg_dup_per_dup_gt:.4f}")

    print("==================================================\n")

    if save_csv_path is not None:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

        fieldnames = [
            "image_id",
            "gt_index",
            "category_id",
            "gt_bbox",
            "matched_pred_num",
            "duplicate_pred_num",
            "status",
            "matched_scores",
            "matched_ious",
            "matched_pred_bboxes",
        ]

        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)

        print(f"✅ 一猪多框明细CSV已保存: {save_csv_path}")


def main():
    os.makedirs(os.path.dirname(PRED_JSON_PATH), exist_ok=True)

    model = load_model()

    coco_gt = COCO(GT_JSON_PATH)

    cat_ids = coco_gt.getCatIds()
    print("GT category ids:", cat_ids)

    if len(cat_ids) != 1:
        print("⚠️ 当前代码默认单类别检测，如果是多类别，需要检查category_id映射。")

    # 单类别猪检测，直接使用GT里的类别id，避免category_id不一致导致评估错误
    fixed_category_id = cat_ids[0]

    transforms = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])

    results = []

    img_files = [f['file_name'] for f in coco_gt.dataset['images']]

    for img_file in tqdm(img_files, desc="正在生成预测JSON"):
        img_path = os.path.join(TEST_IMG_DIR, img_file)

        if not os.path.exists(img_path):
            print(f"⚠️ 图片不存在，跳过: {img_path}")
            continue

        im_pil = Image.open(img_path).convert('RGB')

        w, h = im_pil.size

        orig_size = torch.tensor([[w, h]]).to(DEVICE)

        im_data = transforms(im_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        image_id = get_image_id(img_file, coco_gt)

        if image_id == -1:
            print(f"⚠️ 未找到image_id，跳过: {img_file}")
            continue

        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        labels = labels[0].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < CONF_THRESH:
                continue

            x1, y1, x2, y2 = box

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            results.append({
                "image_id": int(image_id),

                # 单类别检测推荐这样写，保证和test.json一致
                "category_id": int(fixed_category_id),

                "bbox": [
                    float(x1),
                    float(y1),
                    float(bw),
                    float(bh)
                ],
                "score": float(score)
            })

    with open(PRED_JSON_PATH, 'w', encoding="utf-8") as f:
        json.dump(results, f)

    print(f"\n✅ 预测JSON已保存：{PRED_JSON_PATH}")
    print(f"预测框总数: {len(results)}")

    # ====================== 一猪多框统计 ======================
    analyze_duplicate_pig_boxes(
        coco_gt=coco_gt,
        pred_results=results,
        conf_thr=DUP_CONF_THRESH,
        iou_thr=DUP_IOU_THR,
        save_csv_path=DUP_CSV_PATH
    )
    # =========================================================

    print("\n📊 开始计算论文标准COCO指标...")

    coco_pred = coco_gt.loadRes(PRED_JSON_PATH)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n🎉 论文标准测试完成！")


if __name__ == '__main__':
    main()