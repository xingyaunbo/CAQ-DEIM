# -*- coding: utf-8 -*-
"""
根据 Low / Medium / High 分组 json 复制对应图片，
并生成带 GT 框的可视化图片。

用途：
1. 方便人工查看不同遮挡等级样本；
2. 给论文补充 Low / Medium / High 示例图；
3. 不影响原始数据集和 COCO 评估。
"""

import os
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ====================== 原始图片目录 ======================
TEST_IMG_DIR = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\test"


# ====================== 分组 json 路径 ======================
GROUP_JSONS = {
    "Low": r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups\test_easy.json",
    "Medium": r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups\test_medium.json",
    "High": r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups\test_hard.json",
}


# ====================== 输出目录 ======================
OUT_ROOT = r"C:\Users\26930\Desktop\DEIM-main\datasets\visdrone\annotations\occlusion_groups_images"


# 是否复制所有图片
COPY_ALL_IMAGES = True

# 如果只想每组复制前 N 张，设置为数字，比如 20
# 如果 COPY_ALL_IMAGES=True，这个参数无效
MAX_IMAGES_PER_GROUP = 30


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_font(size=18):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\times.ttf",
        "arial.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass

    return ImageFont.load_default()


FONT = load_font(16)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def draw_gt_boxes(image_path, anns, save_path, group_name):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ann in anns:
        if ann.get("iscrowd", 0) == 1:
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])

        # GT 框：绿色
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 220, 0),
            width=3
        )

    # 左上角写组别和 GT 数量
    text = f"{group_name} | GT: {len(anns)}"
    draw.rectangle([5, 5, 190, 32], fill=(255, 255, 255))
    draw.text((10, 8), text, fill=(0, 0, 0), font=FONT)

    img.save(save_path, quality=95)


def process_group(group_name, json_path):
    print(f"\n处理分组：{group_name}")
    print(f"JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    img_dir = mkdir(os.path.join(OUT_ROOT, group_name, "original"))
    gt_dir = mkdir(os.path.join(OUT_ROOT, group_name, "gt"))

    image_id_to_anns = {}
    for ann in dataset["annotations"]:
        image_id = ann["image_id"]
        image_id_to_anns.setdefault(image_id, []).append(ann)

    images = dataset["images"]

    if not COPY_ALL_IMAGES:
        images = images[:MAX_IMAGES_PER_GROUP]

    copied = 0
    missing = 0

    for img_info in images:
        image_id = img_info["id"]
        file_name = img_info["file_name"]

        src_path = os.path.join(TEST_IMG_DIR, file_name)

        if not os.path.exists(src_path):
            print(f"图片不存在，跳过：{src_path}")
            missing += 1
            continue

        stem = Path(file_name).stem
        suffix = Path(file_name).suffix

        # 复制原图
        dst_original = os.path.join(img_dir, f"{stem}{suffix}")
        shutil.copy2(src_path, dst_original)

        # 保存 GT 图
        anns = image_id_to_anns.get(image_id, [])
        dst_gt = os.path.join(gt_dir, f"{stem}_{group_name}_gt.jpg")
        draw_gt_boxes(src_path, anns, dst_gt, group_name)

        copied += 1

    print(f"{group_name} 完成：复制 {copied} 张，缺失 {missing} 张")
    print(f"原图目录：{img_dir}")
    print(f"GT图目录：{gt_dir}")


def main():
    mkdir(OUT_ROOT)

    for group_name, json_path in GROUP_JSONS.items():
        process_group(group_name, json_path)

    print("\n全部完成。")
    print(f"输出目录：{OUT_ROOT}")


if __name__ == "__main__":
    main()