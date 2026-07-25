# -*- coding: utf-8 -*-
"""
PigDetect 论文标准测试脚本：完整测试集 + 遮挡复杂度分组测试

功能：
1. 支持多个模型测试：DEIM baseline / OA-DEIM / final model
2. 先在完整 test.json 上生成预测 JSON
3. 再在 Full / Low / Medium / High 遮挡复杂度子集上分别计算 COCO AP
4. 自动过滤预测 JSON 中不属于当前子集的 image_id
5. 输出每个模型每个分组的 coco_metrics.txt
6. 输出总表 occlusion_group_eval_summary.csv

注意：
- 不需要重新训练
- 不需要重新推理三次，完整测试集推理一次即可
- Low / Medium / High 对应你已经生成的 test_easy.json / test_medium.json / test_hard.json
"""

import os
import sys
import json
import csv
import io
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# =========================================================
# 1. 工程路径
# =========================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


# =========================================================
# 2. 测试集路径
# =========================================================
TEST_IMG_DIR = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\test"

FULL_GT_JSON_PATH = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\test.json"

GROUP_GT_JSONS = {
    "Full": FULL_GT_JSON_PATH,
    "Low": r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups\test_easy.json",
    "Medium": r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups\test_medium.json",
    "High": r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups\test_hard.json",
}


# =========================================================
# 3. 模型配置
#    你可以只保留一个，也可以同时测 baseline 和 final model
# =========================================================
MODEL_CONFIGS = [
    {
        "name": "DEIM",
        "cfg": r"C:\Users\26930\Desktop\DEIM-main\configs\deim_dfine\deim_hgnetv2_n_coco.yml",
        "weight": r"C:\Users\26930\Desktop\DEIM-main\deim_outputs\deim_hgnetv2_n_coco\baseline1.pth",
    },

    # 把下面路径改成你的最终模型 cfg 和 pth
    {
        "name": "OA-DEIM",
        "cfg": r"C:\Users\26930\Desktop\DEIM-main\configs\deim_dfine\deim_hgnetv2_n_coco.yml",
        "weight": r"C:\Users\26930\Desktop\DEIM-main\deim_outputs\deim_hgnetv2_n_coco\best_stg3.pth",
    },
]


# =========================================================
# 4. 输出路径
# =========================================================
OUT_ROOT = r"C:\Users\26930\Desktop\DEIM-main\predict_results\occlusion_group_eval"
os.makedirs(OUT_ROOT, exist_ok=True)


# =========================================================
# 5. 测试参数
# =========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (960, 960)     # h, w

# COCO 标准测试建议保持很低
CONF_THRESH = 0.001

# 单类别 pig 数据集建议 True
# 这样 category_id 会自动使用 GT json 里的类别 id，避免 label=0 / category_id=1 不匹配
FORCE_SINGLE_CLASS_CATEGORY_ID = True


# =========================================================
# 6. 工具函数
# =========================================================
def get_image_id(filename, coco_gt):
    for img_info in coco_gt.dataset["images"]:
        if img_info["file_name"] == filename:
            return img_info["id"]
    return -1


def safe_load_state_dict(cfg_model, state, model_name):
    if isinstance(state, dict):
        keys = list(state.keys())
        if len(keys) > 0 and all(k.startswith("module.") for k in keys):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    incompatible = cfg_model.load_state_dict(state, strict=False)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    if len(missing) > 0:
        print(f"[{model_name}] missing keys 数量：{len(missing)}，前 10 个：")
        print(missing[:10])

    if len(unexpected) > 0:
        print(f"[{model_name}] unexpected keys 数量：{len(unexpected)}，前 10 个：")
        print(unexpected[:10])


class DeployModel(nn.Module):
    def __init__(self, model_name, cfg_path, weight_path):
        super().__init__()

        print("\n==============================")
        print(f"加载模型：{model_name}")
        print(f"CFG: {cfg_path}")
        print(f"WEIGHT: {weight_path}")
        print("==============================")

        cfg = YAMLConfig(cfg_path, resume=weight_path)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(weight_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "ema" in checkpoint:
            ema = checkpoint["ema"]
            if isinstance(ema, dict) and "module" in ema:
                state = ema["module"]
            else:
                state = ema
            print(f"[{model_name}] 使用 EMA 权重")
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
            print(f"[{model_name}] 使用 model 权重")
        else:
            state = checkpoint
            print(f"[{model_name}] 使用原始 state_dict 权重")

        safe_load_state_dict(cfg.model, state, model_name)

        self.model = cfg.model.deploy().to(DEVICE)
        self.model.eval()

        self.postprocessor = cfg.postprocessor.deploy()
        if isinstance(self.postprocessor, nn.Module):
            self.postprocessor = self.postprocessor.to(DEVICE)
            self.postprocessor.eval()

    @torch.no_grad()
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


def generate_predictions(model, model_name, coco_gt, cat_ids, save_json_path):
    transforms = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
    ])

    results = []
    img_files = [f["file_name"] for f in coco_gt.dataset["images"]]

    for img_file in tqdm(img_files, desc=f"{model_name} 正在生成完整测试集预测 JSON"):
        img_path = os.path.join(TEST_IMG_DIR, img_file)

        if not os.path.exists(img_path):
            print(f"图片不存在，跳过：{img_path}")
            continue

        image_id = get_image_id(img_file, coco_gt)
        if image_id == -1:
            continue

        im_pil = Image.open(img_path).convert("RGB")
        w, h = im_pil.size

        orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(DEVICE)
        im_data = transforms(im_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        boxes = boxes[0].detach().cpu().numpy()
        scores = scores[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if float(score) < CONF_THRESH:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            if FORCE_SINGLE_CLASS_CATEGORY_ID and len(cat_ids) == 1:
                category_id = int(cat_ids[0])
            else:
                category_id = int(label)

            results.append({
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": [x1, y1, bw, bh],
                "score": float(score),
            })

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"\n✅ {model_name} 完整测试集预测 JSON 已保存：{save_json_path}")
    print(f"预测框数量：{len(results)}")

    return save_json_path


def filter_predictions_by_group(gt_json_path, pred_json_path, save_path):
    coco_gt = COCO(gt_json_path)
    valid_img_ids = set(coco_gt.getImgIds())

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    filtered = [p for p in preds if int(p["image_id"]) in valid_img_ids]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f)

    return save_path, len(filtered)


def run_coco_eval(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    if len(preds) == 0:
        metrics = {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }
        return metrics, "No predictions."

    coco_pred = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.catIds = coco_gt.getCatIds()

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    metric_text = output.getvalue()
    s = coco_eval.stats

    metrics = {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR1": float(s[6]),
        "AR10": float(s[7]),
        "AR100": float(s[8]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
    }

    return metrics, metric_text


def evaluate_all_groups(model_name, full_pred_json_path):
    rows = []

    model_out_dir = os.path.join(OUT_ROOT, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    for group_name, gt_path in GROUP_GT_JSONS.items():
        print("\n==============================")
        print(f"评估模型：{model_name}")
        print(f"分组：{group_name}")
        print(f"GT: {gt_path}")
        print("==============================")

        coco_gt = COCO(gt_path)
        num_images = len(coco_gt.getImgIds())
        num_anns = len(coco_gt.getAnnIds())

        filtered_pred_path = os.path.join(
            model_out_dir,
            f"{model_name}_{group_name}_filtered_predictions.json"
        )

        filtered_pred_path, num_preds = filter_predictions_by_group(
            gt_path,
            full_pred_json_path,
            filtered_pred_path,
        )

        metrics, metric_text = run_coco_eval(gt_path, filtered_pred_path)

        txt_path = os.path.join(
            model_out_dir,
            f"{model_name}_{group_name}_coco_metrics.txt"
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(metric_text)

        print(metric_text)

        row = {
            "Model": model_name,
            "Group": group_name,
            "Images": num_images,
            "Annotations": num_anns,
            "Predictions": num_preds,
            **metrics,
        }

        rows.append(row)

    return rows


def save_summary_csv(rows):
    save_path = os.path.join(OUT_ROOT, "occlusion_group_eval_summary.csv")

    fieldnames = [
        "Model",
        "Group",
        "Images",
        "Annotations",
        "Predictions",
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\n✅ 分组评估汇总表已保存：{save_path}")


def main():
    print(f"使用设备：{DEVICE}")
    print(f"输出目录：{OUT_ROOT}")

    full_coco_gt = COCO(FULL_GT_JSON_PATH)
    cat_ids = full_coco_gt.getCatIds()

    all_rows = []

    for cfg in MODEL_CONFIGS:
        model_name = cfg["name"]
        cfg_path = cfg["cfg"]
        weight_path = cfg["weight"]

        model_out_dir = os.path.join(OUT_ROOT, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        full_pred_json_path = os.path.join(
            model_out_dir,
            f"{model_name}_full_test_predictions.json"
        )

        model = DeployModel(model_name, cfg_path, weight_path).to(DEVICE)
        model.eval()

        generate_predictions(
            model=model,
            model_name=model_name,
            coco_gt=full_coco_gt,
            cat_ids=cat_ids,
            save_json_path=full_pred_json_path,
        )

        rows = evaluate_all_groups(model_name, full_pred_json_path)
        all_rows.extend(rows)

        del model
        torch.cuda.empty_cache()

    save_summary_csv(all_rows)

    print("\n🎉 全部测试完成！")
    print("重点查看：")
    print(os.path.join(OUT_ROOT, "occlusion_group_eval_summary.csv"))


if __name__ == "__main__":
    main()