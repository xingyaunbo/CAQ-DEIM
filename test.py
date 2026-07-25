"""
论文标准测试脚本 - 修正版
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import sys
import os
import json
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# ====================== 论文测试固定配置 ======================
CFG_PATH = r"C:\Users\26930\Desktop\DEIM-main\configs\deim_dfine\deim_hgnetv2_n_coco.yml"
WEIGHT_PATH = r"C:\Users\26930\Desktop\DEIM-main\deim_outputs\deim_hgnetv2_n_coco\best_stg2.pth"
TEST_IMG_DIR = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\val"
GT_JSON_PATH = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\val.json"
PRED_JSON_PATH = r"C:\Users\26930\Desktop\DEIM-main\predict_results\coco_predictions.json"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.001
# ==============================================================

def load_model():
    cfg = YAMLConfig(CFG_PATH, resume=WEIGHT_PATH)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
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
    print(f"✅ 模型加载完成（EMA权重）")
    return model

def get_image_id(filename, coco_gt):
    for img_info in coco_gt.dataset['images']:
        if img_info['file_name'] == filename:
            return img_info['id']
    return -1

def main():
    os.makedirs(os.path.dirname(PRED_JSON_PATH), exist_ok=True)
    model = load_model()
    coco_gt = COCO(GT_JSON_PATH)

    transforms = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])

    results = []
    img_files = [f['file_name'] for f in coco_gt.dataset['images']]

    for img_file in tqdm(img_files, desc="正在生成预测JSON"):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        if not os.path.exists(img_path):
            continue

        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(DEVICE)
        im_data = transforms(im_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        image_id = get_image_id(img_file, coco_gt)
        if image_id == -1:
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

            results.append({
                "image_id": int(image_id),
                "category_id": int(label),  # 🔴 修正：改成从0开始
                "bbox": [float(x1), float(y1), float(bw), float(bh)],
                "score": float(score)
            })

    with open(PRED_JSON_PATH, 'w') as f:
        json.dump(results, f)
    print(f"\n✅ 预测JSON已保存：{PRED_JSON_PATH}")

    print("\n📊 开始计算论文标准COCO指标...")
    coco_pred = coco_gt.loadRes(PRED_JSON_PATH)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n🎉 论文标准测试完成！")

if __name__ == '__main__':
    main()