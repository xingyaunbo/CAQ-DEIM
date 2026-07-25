"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import json
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four selected images
    into a single composite image with randomized transformations.

    Original:
        current image + 3 random images

    Modified when score_file is provided:
        current image + high-occlusion image + mid-occlusion image + random image

    If score_file is None, this class behaves like the original Mosaic.
    """

    def __init__(
        self,
        output_size=320,
        max_size=None,
        rotation_range=0,
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),
        probability=1.0,
        fill_value=114,
        use_cache=True,
        max_cached_images=50,
        random_pop=True,

        # Occlusion-aware sampling parameters
        score_file=None,
        high_thr=0.5,
        mid_thr=0.25,
        high_prob=0.6,
        mid_prob=0.4,
    ) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.

            score_file (str | None): JSON file containing occlusion/density score of each image.
            high_thr (float): Score threshold for high-occlusion images.
            mid_thr (float): Score threshold for medium-occlusion images.
            high_prob (float): Probability of sampling one image from high-occlusion pool.
            mid_prob (float): Probability of sampling one image from medium-occlusion pool.
        """
        super().__init__()

        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(
            degrees=rotation_range,
            translate=translation_range,
            scale=scaling_range,
            fill=fill_value,
        )

        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

        # Occlusion-aware sampling config
        self.score_file = score_file
        self.high_thr = high_thr
        self.mid_thr = mid_thr
        self.high_prob = high_prob
        self.mid_prob = mid_prob

        self.score_by_imgid = {}
        self.high_indices = []
        self.mid_indices = []
        self.all_indices = []
        self._score_pools_built = False

        if score_file is not None:
            with open(score_file, "r", encoding="utf-8") as f:
                score_data = json.load(f)

            self.score_by_imgid = {
                int(k): float(v["score"]) for k, v in score_data.items()
            }

            print(
                f"[Mosaic] Occlusion-aware sampling enabled. "
                f"score_file={score_file}, "
                f"high_thr={high_thr}, mid_thr={mid_thr}, "
                f"high_prob={high_prob}, mid_prob={mid_prob}"
            )

            if use_cache:
                print(
                    "[Mosaic] Warning: use_cache=True. "
                    "Occlusion-aware sampling only works in dataset sampling branch. "
                    "Set use_cache=False if you want score-aware sampling to take effect."
                )

    def _build_score_pools(self, dataset):
        """
        Build image index pools according to occlusion score.

        COCO dataset usually has:
            dataset.ids[index] = image_id
        """
        if self._score_pools_built:
            return

        self.all_indices = list(range(len(dataset)))
        self.high_indices = []
        self.mid_indices = []

        if self.score_file is None or not hasattr(dataset, "ids"):
            self._score_pools_built = True
            return

        for idx, img_id in enumerate(dataset.ids):
            img_id = int(img_id)
            score = self.score_by_imgid.get(img_id, 0.0)

            if score >= self.high_thr:
                self.high_indices.append(idx)
            elif score >= self.mid_thr:
                self.mid_indices.append(idx)

        self._score_pools_built = True

        print(
            f"[Mosaic] Occlusion-aware pools built: "
            f"high={len(self.high_indices)}, "
            f"mid={len(self.mid_indices)}, "
            f"all={len(self.all_indices)}"
        )

    def _sample_from_pool(self, pool_name):
        if pool_name == "high" and len(self.high_indices) > 0:
            return random.choice(self.high_indices)

        if pool_name == "mid" and len(self.mid_indices) > 0:
            return random.choice(self.mid_indices)

        return random.choice(self.all_indices)

    def _sample_extra_indices(self, dataset):
        """
        Original Mosaic:
            random.choices(range(len(dataset)), k=3)

        Modified Mosaic:
            one high-occlusion image + one medium-occlusion image + one random image.
        """
        if self.score_file is None:
            return random.choices(range(len(dataset)), k=3)

        self._build_score_pools(dataset)

        sample_indices = []

        # Image 2: prefer high-occlusion / high-density image
        if random.random() < self.high_prob:
            sample_indices.append(self._sample_from_pool("high"))
        else:
            sample_indices.append(self._sample_from_pool("random"))

        # Image 3: prefer medium-occlusion / medium-density image
        if random.random() < self.mid_prob:
            sample_indices.append(self._sample_from_pool("mid"))
        else:
            sample_indices.append(self._sample_from_pool("random"))

        # Image 4: random image to keep original data distribution
        sample_indices.append(self._sample_from_pool("random"))

        return sample_indices

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size

        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # Original:
        # sample_indices = random.choices(range(len(dataset)), k=3)
        #
        # Modified:
        # sample images according to occlusion/density score when score_file is provided.
        sample_indices = self._sample_extra_indices(dataset)

        for idx in sample_indices:
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)

        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [
            dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"]))
            for idx in sample_indices
        ]
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [
            [0, 0],
            [max_width, 0],
            [0, max_height],
            [max_width, max_height],
        ]

        merged_image = Image.new(
            mode=mosaic_samples[0]["img"].mode,
            size=(max_width * 2, max_height * 2),
            color=0,
        )

        offsets = torch.tensor(
            [
                [0, 0],
                [max_width, 0],
                [0, max_height],
                [max_width, max_height],
            ]
        ).repeat(1, 2)

        mosaic_target = []

        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            target["boxes"] = target["boxes"] + offsets[i]
            mosaic_target.append(target)

        merged_target = {}
        for key in mosaic_target[0]:
            merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
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

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor(
            [
                [0, 0],
                [max_width, 0],
                [0, max_height],
                [max_width, max_height],
            ]
        ).repeat(1, 2)

        merged_target = {}

        for key in targets[0]:
            if key == "boxes":
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(
                image,
                target,
                self.mosaic_cache,
            )
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(
                mosaic_samples,
                max_height,
                max_width,
            )
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(
                image,
                target,
                dataset,
            )
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(
                resized_images,
                resized_targets,
                max_height,
                max_width,
            )

        # Clamp boxes and convert target formats
        if "boxes" in mosaic_target:
            mosaic_target["boxes"] = convert_to_tv_tensor(
                mosaic_target["boxes"],
                "boxes",
                box_format="xyxy",
                spatial_size=mosaic_image.size[::-1],
            )

        if "masks" in mosaic_target:
            mosaic_target["masks"] = convert_to_tv_tensor(
                mosaic_target["masks"],
                "masks",
            )

        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(
            mosaic_image,
            mosaic_target,
        )

        return mosaic_image, mosaic_target, dataset