"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 mal_alpha=None,
                 use_uni_set=True,

                 # AAM-Loss
                 amal_lambda=0.0,
                 amal_min_target=0.0,
                 oal_alpha=0.0,
                 oal_max_w=1.0,

                 # Duplicate Query Suppression
                 dup_iou_thr=0.75,
                 dup_pred_iou_thr=0.70,
                 dup_conf_thr=0.35,
                 dup_beta=3.0,
                 dup_gamma=2.0,
                 hq_mal_thr=0.55,
                 hq_mal_eta=0.25,
                 ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        # AAM-Loss parameters
        self.amal_lambda = amal_lambda
        self.amal_min_target = amal_min_target
        self.oal_alpha = oal_alpha
        self.oal_max_w = oal_max_w

        # DQS parameters
        self.dup_iou_thr = dup_iou_thr
        self.dup_pred_iou_thr = dup_pred_iou_thr
        self.dup_conf_thr = dup_conf_thr
        self.dup_beta = dup_beta
        self.dup_gamma = dup_gamma
        self.hq_mal_thr = hq_mal_thr
        self.hq_mal_eta = hq_mal_eta

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """
        Adhesion-aware MAL.

        原 MAL:
            target_score = IoU ^ gamma

        A-MAL:
            adhesion_i = max IoU between current GT and other GTs
            gamma_i = gamma / (1 + lambda * adhesion_i)
            target_score = IoU ^ gamma_i
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )

            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
            ious = torch.diag(ious).detach()
        else:
            ious = values.detach()

        src_logits = outputs['pred_logits']

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(
            target_classes,
            num_classes=self.num_classes + 1
        )[..., :-1]

        # ================= A-MAL 核心改动 =================
        target_score_o = torch.zeros_like(
            target_classes,
            dtype=src_logits.dtype
        )

        if ious.numel() > 0:
            adhesion = self._get_matched_adhesion(targets, indices).to(
                device=ious.device,
                dtype=ious.dtype
            )

            if adhesion.numel() != ious.numel():
                adhesion = torch.zeros_like(ious)

            # gamma_i = gamma / (1 + lambda * adhesion)
            gamma_i = self.gamma / (1.0 + self.amal_lambda * adhesion)
            gamma_i = gamma_i.clamp(min=0.5, max=self.gamma)

            q = ious.clamp(0.0, 1.0)

            base_score = q.pow(self.gamma)

            hq_mask = (q >= self.hq_mal_thr).to(q.dtype)

            matched_score = base_score + self.hq_mal_eta * hq_mask * (q - base_score)

            matched_score = matched_score.clamp(0.0, 1.0)

            if self.amal_min_target > 0:
                min_score = self.amal_min_target * adhesion
                matched_score = torch.maximum(matched_score, min_score)

            target_score_o[idx] = matched_score.to(target_score_o.dtype)

        target_score = target_score_o.unsqueeze(-1) * target
        # ==================================================

        pred_score = F.sigmoid(src_logits).detach()

        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(
            src_logits,
            target_score,
            weight=weight,
            reduction='none'
        )

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """
        Quality-gated Overlap-aware Localization Reweighting.

        原始 OAL:
            w = 1 + alpha * adhesion

        改进后:
            w = 1 + alpha * adhesion * (1 - IoU)

        含义:
            只有“粘连程度高 + 当前定位质量差”的目标才会被加强；
            如果粘连目标已经定位较准，则不再过度加权。
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        losses = {}

        if src_boxes.numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            losses['loss_bbox'] = zero
            losses['loss_giou'] = zero
            return losses

        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

        # matched prediction 与 matched GT 的 IoU
        # 这里 detach，只作为权重，不让权重本身参与反向传播
        with torch.no_grad():
            pair_iou, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
            matched_iou = torch.diag(pair_iou).clamp(0.0, 1.0)

        # 每个 matched GT 的粘连程度
        adhesion = self._get_matched_adhesion(targets, indices).to(
            device=src_boxes.device,
            dtype=src_boxes.dtype
        )

        if adhesion.numel() != src_boxes.shape[0]:
            adhesion = torch.zeros(
                src_boxes.shape[0],
                device=src_boxes.device,
                dtype=src_boxes.dtype
            )

        # ================= Quality-gated OAL 核心 =================
        # 粘连越严重、当前 IoU 越低，定位权重越高
        quality_gap = (1.0 - matched_iou).to(src_boxes.dtype)

        loc_weight = 1.0 + self.oal_alpha * adhesion * quality_gap
        loc_weight = loc_weight.clamp(max=self.oal_max_w).detach()
        # ==========================================================

        # L1 bbox loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum(dim=-1)

        if boxes_weight is not None:
            loc_weight = loc_weight * boxes_weight.to(loc_weight.dtype)

        losses['loss_bbox'] = (loss_bbox * loc_weight).sum() / num_boxes

        # GIoU loss
        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
        )

        losses['loss_giou'] = (loss_giou * loc_weight).sum() / num_boxes

        return losses

    def loss_dup(self, outputs, targets, indices, num_boxes):
        """
        Duplicate Query Suppression.

        只抑制：
        1. 未匹配 query；
        2. 与某个 GT 高 IoU；
        3. 置信度较高；
        4. 与该 GT 的 matched query 预测框也高度重叠。

        这样比单纯 pred-GT IoU 抑制更安全。
        """
        assert 'pred_boxes' in outputs
        assert 'pred_logits' in outputs

        # DN 分支不做重复框抑制
        if 'is_dn' in outputs:
            zero = outputs['pred_boxes'].sum() * 0.0
            return {'loss_dup': zero}

        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']
        device = pred_boxes.device

        total_loss = pred_boxes.sum() * 0.0

        bs, num_queries = pred_boxes.shape[:2]

        for b in range(bs):
            gt_boxes = targets[b]['boxes']

            if gt_boxes.numel() == 0:
                continue

            src_idx, tgt_idx = indices[b]

            if len(src_idx) == 0:
                continue

            boxes_b = pred_boxes[b]
            scores_b = pred_logits[b].sigmoid().max(dim=-1)[0]

            unmatched = torch.ones(
                num_queries,
                dtype=torch.bool,
                device=device
            )
            unmatched[src_idx] = False

            # query 与所有 GT 的最大 IoU
            iou_q_gt, _ = box_iou(
                box_cxcywh_to_xyxy(boxes_b),
                box_cxcywh_to_xyxy(gt_boxes)
            )

            max_iou, max_gt = iou_q_gt.max(dim=1)

            cand = (
                    unmatched
                    & (max_iou > self.dup_iou_thr)
                    & (scores_b > self.dup_conf_thr)
            )

            if cand.sum() == 0:
                continue

            # 建立 gt_idx -> matched query idx 映射
            matched_for_gt = torch.full(
                (gt_boxes.shape[0],),
                -1,
                dtype=torch.long,
                device=device
            )
            matched_for_gt[tgt_idx] = src_idx

            cand_idx = torch.nonzero(cand).squeeze(1)
            cand_gt = max_gt[cand_idx]
            matched_q = matched_for_gt[cand_gt]

            valid = matched_q >= 0

            if valid.sum() == 0:
                continue

            cand_idx = cand_idx[valid]
            matched_q = matched_q[valid]

            # 候选重复框与 matched query 预测框之间的 IoU
            iou_pred_pred, _ = box_iou(
                box_cxcywh_to_xyxy(boxes_b[cand_idx]),
                box_cxcywh_to_xyxy(boxes_b[matched_q])
            )
            pred_pair_iou = torch.diag(iou_pred_pred)

            valid2 = pred_pair_iou > self.dup_pred_iou_thr

            if valid2.sum() == 0:
                continue

            cand_idx = cand_idx[valid2]
            pred_pair_iou = pred_pair_iou[valid2]
            q_gt_iou = max_iou[cand_idx]
            q_score = scores_b[cand_idx]

            iou_weight = (
                    (q_gt_iou - self.dup_iou_thr)
                    / max(1e-6, 1.0 - self.dup_iou_thr)
            ).clamp(min=0.0).pow(self.dup_beta)

            pred_weight = (
                    (pred_pair_iou - self.dup_pred_iou_thr)
                    / max(1e-6, 1.0 - self.dup_pred_iou_thr)
            ).clamp(min=0.0)

            score_penalty = (
                    q_score - self.dup_conf_thr
            ).clamp(min=0.0).pow(self.dup_gamma)

            total_loss = total_loss + (iou_weight * pred_weight * score_penalty).sum()

        total_loss = total_loss / num_boxes

        return {'loss_dup': total_loss}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_target_adhesion_all(self, targets):
        """
        计算每张图中每个 GT 的粘连程度。
        adhesion = 当前 GT 与同图其他 GT 的最大 IoU。
        返回:
            List[Tensor], 每个 tensor shape = [num_gt]
        """
        adhesion_list = []

        for t in targets:
            boxes = t["boxes"]
            device = boxes.device

            if boxes.numel() == 0:
                adhesion_list.append(torch.zeros(0, device=device))
                continue

            if boxes.shape[0] == 1:
                adhesion_list.append(torch.zeros(1, device=device))
                continue

            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            ious, _ = box_iou(boxes_xyxy, boxes_xyxy)

            # 去掉自身 IoU
            ious.fill_diagonal_(0.0)

            adhesion = ious.max(dim=1)[0].detach()
            adhesion = adhesion.clamp(0.0, 1.0)
            adhesion_list.append(adhesion)

        return adhesion_list

    def _get_matched_adhesion(self, targets, indices):
        """
        按照 indices 顺序，取出 matched GT 的 adhesion。
        返回:
            Tensor shape = [num_matched]
        """
        adhesion_all = self._get_target_adhesion_all(targets)

        matched_adhesion = []
        for adhesion, (_, tgt_idx) in zip(adhesion_all, indices):
            if len(tgt_idx) == 0:
                continue
            matched_adhesion.append(adhesion[tgt_idx])

        if len(matched_adhesion) == 0:
            device = targets[0]["boxes"].device
            return torch.zeros(0, device=device)

        return torch.cat(matched_adhesion, dim=0)


    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'dup': self.loss_dup,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
