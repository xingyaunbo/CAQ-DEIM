"""
Occlusion-Aware Mosaic for DEIM / D-FINE.

This transform keeps the same YAML interface style as the original Mosaic:
- output_size
- rotation_range
- translation_range
- scaling_range
- probability
- fill_value
- use_cache
- max_cached_images
- random_pop

The difference is that extra images are sampled according to occlusion/density score.
"""

import json
import random

import torch
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register

torchvision.disable_beta_transforms_warning()


@register()
class OcclusionAwareMosaic(T.Transform):
    def __init__(
        self,
        output_size=480,
        max_size=None,
        rotation_range=0,
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),
        probability=1.0,
        fill_value=0,
        score_file=None,
        high_thr=0.7,
        mid_thr=0.3,
        high_prob=0.7,
        mid_prob=0.5,
        use_cache=True,
        max_cached_images=80,
        random_pop=True,
        cache_prob=0.7,
    ) -> None:
        super().__init__()

        self.output_size = output_size
        self.max_size = max_size
        self.probability = probability
        self.score_file = score_file
        self.high_thr = high_thr
        self.mid_thr = mid_thr
        self.high_prob = high_prob
        self.mid_prob = mid_prob

        self.use_cache = use_cache
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.cache_prob = cache_prob
        self.mosaic_cache = []

        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.affine_transform = T.RandomAffine(
            degrees=rotation_range,
            translate=translation_range,
            scale=scaling_range,
            fill=fill_value,
        )

        self.score_by_imgid = {}
        self.high_ids = []
        self.mid_ids = []
        self.all_ids = []

        if score_file is not None:
            with open(score_file, "r", encoding="utf-8") as f:
                score_data = json.load(f)

            self.score_by_imgid = {
                int(k): float(v["score"]) for k, v in score_data.items()
            }

            self.high_ids = [
                img_id for img_id, score in self.score_by_imgid.items()
                if score >= high_thr
            ]
            self.mid_ids = [
                img_id for img_id, score in self.score_by_imgid.items()
                if mid_thr <= score < high_thr
            ]
            self.all_ids = list(self.score_by_imgid.keys())

            print(
                f"[OcclusionAwareMosaic] score_file={score_file}, "
                f"high={len(self.high_ids)}, mid={len(self.mid_ids)}, all={len(self.all_ids)}, "
                f"use_cache={self.use_cache}, cache_prob={self.cache_prob}"
            )
        else:
            print("[OcclusionAwareMosaic] Warning: score_file=None, fallback to random sampling.")

    @staticmethod
    def _clone_target(target):
        out = {}
        for k, v in target.items():
            out[k] = v.clone() if hasattr(v, "clone") else v
        return out

    def _add_to_cache(self, image, target):
        if not self.use_cache:
            return

        self.mosaic_cache.append(
            {
                "image": image.copy(),
                "target": self._clone_target(target),
            }
        )

        if len(self.mosaic_cache) > self.max_cached_images:
            if self.random_pop:
                # do not remove the newest sample
                pop_idx = random.randint(0, len(self.mosaic_cache) - 2)
            else:
                pop_idx = 0
            self.mosaic_cache.pop(pop_idx)

    def _sample_image_id(self, pool_type="random"):
        if pool_type == "high" and len(self.high_ids) > 0:
            return random.choice(self.high_ids)

        if pool_type == "mid" and len(self.mid_ids) > 0:
            return random.choice(self.mid_ids)

        if len(self.all_ids) > 0:
            return random.choice(self.all_ids)

        return None

    def _image_id_to_dataset_index(self, dataset, image_id):
        if image_id is None:
            return random.randrange(len(dataset))

        if hasattr(dataset, "ids"):
            try:
                return dataset.ids.index(image_id)
            except ValueError:
                return random.randrange(len(dataset))

        return random.randrange(len(dataset))

    def _load_by_image_id(self, dataset, image_id):
        idx = self._image_id_to_dataset_index(dataset, image_id)
        image, target = dataset.load_item(idx)
        image, target = self.resize(image, target)
        return image, target

    def _load_from_cache_or_dataset(self, dataset, image_id):
        if (
            self.use_cache
            and len(self.mosaic_cache) >= 4
            and random.random() < self.cache_prob
        ):
            sample = random.choice(self.mosaic_cache)
            return sample["image"].copy(), self._clone_target(sample["target"])

        return self._load_by_image_id(dataset, image_id)

    def _load_samples(self, image, target, dataset):
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size

        # Current image has already been resized in forward.
        images = [image]
        targets = [target]

        max_height, max_width = get_size_func(image)

        # Image 2: prefer high-occlusion / high-density image.
        if random.random() < self.high_prob:
            img_id = self._sample_image_id("high")
        else:
            img_id = self._sample_image_id("random")

        img, tgt = self._load_from_cache_or_dataset(dataset, img_id)
        h, w = get_size_func(img)
        max_height, max_width = max(max_height, h), max(max_width, w)
        images.append(img)
        targets.append(tgt)

        # Image 3: prefer medium-occlusion / medium-density image.
        if random.random() < self.mid_prob:
            img_id = self._sample_image_id("mid")
        else:
            img_id = self._sample_image_id("random")

        img, tgt = self._load_from_cache_or_dataset(dataset, img_id)
        h, w = get_size_func(img)
        max_height, max_width = max(max_height, h), max(max_width, w)
        images.append(img)
        targets.append(tgt)

        # Image 4: random image to keep global distribution.
        img_id = self._sample_image_id("random")
        img, tgt = self._load_from_cache_or_dataset(dataset, img_id)
        h, w = get_size_func(img)
        max_height, max_width = max(max_height, h), max(max_width, w)
        images.append(img)
        targets.append(tgt)

        return images, targets, max_height, max_width

    def _create_mosaic(self, images, targets, max_height, max_width):
        placement_offsets = [
            [0, 0],
            [max_width, 0],
            [0, max_height],
            [max_width, max_height],
        ]

        merged_image = Image.new(
            mode=images[0].mode,
            size=(max_width * 2, max_height * 2),
            color=0,
        )

        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        offsets = torch.tensor(
            [
                [0, 0],
                [max_width, 0],
                [0, max_height],
                [max_width, max_height],
            ],
            dtype=torch.float32,
        ).repeat(1, 2)

        merged_target = {}

        for key in targets[0]:
            if key in ["orig_size", "size"]:
                continue

            values = []

            for i, target in enumerate(targets):
                if key not in target:
                    continue

                if key == "boxes":
                    values.append(target[key] + offsets[i])
                else:
                    values.append(target[key])

            if len(values) == 0:
                continue

            if isinstance(values[0], torch.Tensor):
                try:
                    merged_target[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    # Skip keys that cannot be concatenated safely.
                    pass
            else:
                merged_target[key] = values

        return merged_image, merged_target

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        # Match original Mosaic behavior: resize the current image first.
        image, target = self.resize(image, target)

        # Cache resized current sample.
        self._add_to_cache(image, target)

        # Skip mosaic according to probability.
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        images, targets, max_height, max_width = self._load_samples(
            image, target, dataset
        )

        mosaic_image, mosaic_target = self._create_mosaic(
            images,
            targets,
            max_height,
            max_width,
        )

        if "boxes" in mosaic_target:
            img_w, img_h = mosaic_image.size
            boxes = mosaic_target["boxes"]

            boxes[:, 0::2].clamp_(min=0, max=img_w)
            boxes[:, 1::2].clamp_(min=0, max=img_h)

            mosaic_target["boxes"] = convert_to_tv_tensor(
                boxes,
                "boxes",
                box_format="xyxy",
                spatial_size=mosaic_image.size[::-1],
            )

        if "masks" in mosaic_target:
            mosaic_target["masks"] = convert_to_tv_tensor(
                mosaic_target["masks"],
                "masks",
            )

        if "labels" not in mosaic_target or "boxes" not in mosaic_target:
            return image, target, dataset

        mosaic_image, mosaic_target = self.affine_transform(
            mosaic_image,
            mosaic_target,
        )

        return mosaic_image, mosaic_target, dataset