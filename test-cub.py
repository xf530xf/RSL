"""Weakly-supervised object-localization evaluation for the CUB-200-2011 pipeline.

Mirrors `test-underwater.py` (multi-scale + horizontal-flip test-time
augmentation), but the metric is bounding-box localization against CUB's
`bounding_boxes.txt`:

    * GT-Known Loc : fraction of images whose predicted box has IoU >= 0.5 with
      the GT box (using the ground-truth class to derive the CAM).
    * mean IoU     : average IoU between predicted and GT box.

Two predicted boxes are reported: one from the segmentation foreground map
(``1 - p(background)``) and one from the CAM pseudo-label.
"""

import argparse
import os
import sys

sys.path.append('.')

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import cub
from utils.loc import bbox_iou, normalize_scoremap, scoremap_to_bbox
from RSL.model_attn_aff_cub import RSL


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/cub_attn_reg.yaml', type=str)
parser.add_argument('--work_dir', default='results/cub', type=str)
parser.add_argument('--eval_set', default='val', type=str)
parser.add_argument('--model_path', default='path/to/rsl_iter_20000.pth', type=str)
parser.add_argument('--scales', default='1,1.25', type=str,
                    help='comma separated test-time scales (besides flip)')
parser.add_argument('--save_vis', action='store_true', help='save box visualizations')


def _round_to_flag(inputs, flag):
    _, _, h, w = inputs.shape
    new_h = max(flag, round(h / flag) * flag)
    new_w = max(flag, round(w / flag) * flag)
    if (new_h, new_w) != (h, w):
        inputs = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return inputs


def validate(model, dataset, cfg, test_scales=(1.0,)):
    # NOTE: no torch.no_grad() — the model's internal CLIP GradCAM needs
    # gradients.  Outputs are detached before numpy conversion below.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
    )
    model.cuda()
    model.eval()

    flag = cfg.clip_init.clip_flag
    resize_long = cfg.clip_init.resize_long
    cam_thr = float(getattr(cfg.eval, 'cam_threshold', 0.5))
    iou_thr = float(getattr(cfg.eval, 'iou_threshold', 0.5))

    n_total = 0
    # classification
    cls_top1_sum, cls_top5_sum = 0, 0
    # GT-Known Loc (box only)  +  Top-1/Top-5 Loc (cls AND box)
    gtk_seg_sum, gtk_cam_sum = 0, 0
    top1_loc_seg_sum, top5_loc_seg_sum = 0, 0
    top1_loc_cam_sum, top5_loc_cam_sum = 0, 0
    seg_iou_sum, cam_iou_sum = 0.0, 0.0

    for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
        name, inputs, gt_box, cls_label, size = data
        names = list(name) + list(name)
        cls_labels_cat = torch.cat([cls_label, cls_label], dim=0)

        inputs = inputs.cuda()
        W, H = int(size[0, 0]), int(size[0, 1])
        gt = np.asarray(gt_box[0].numpy(), dtype=np.float32)

        # ground-truth species id in model space (1..200) -> 0-indexed 0..199
        gt_cls = int(np.nonzero(np.asarray(cls_label[0]) > 0)[0][0]) - 1

        _, _, h, w = inputs.shape
        ratio = resize_long / max(h, w)
        base = F.interpolate(inputs, size=(int(h * ratio), int(w * ratio)),
                             mode='bilinear', align_corners=False)

        # ---- classification (zero-shot CLIP over the 200 species) ----
        with torch.no_grad():
            cls_prob = model.classify(_round_to_flag(base, flag))[0]
        top5 = torch.topk(cls_prob, 5).indices.detach().cpu().numpy().tolist()
        cls_top1 = int(top5[0] == gt_cls)
        cls_top5 = int(gt_cls in top5)

        fg_maps = []
        cam_unflipped = None
        for s in test_scales:
            scaled = base if s == 1.0 else F.interpolate(base, scale_factor=s,
                                                         mode='bilinear', align_corners=False)
            scaled = _round_to_flag(scaled, flag)
            inputs_cat = torch.cat([scaled, scaled.flip(-1)], dim=0)

            segs_clip_cat, segs_dino_cat, cam, _, cam_raw_list = model(
                inputs_cat, names, cls_labels=cls_labels_cat, mode='val',
            )
            segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat
            segs_up = F.interpolate(segs_cat, size=(H, W), mode='bilinear', align_corners=False)
            prob = F.softmax(segs_up, dim=1)
            fg = 1.0 - prob[:, 0]                       # (2, H, W)
            fg = 0.5 * (fg[0] + fg[1].flip(-1))         # un-flip + average
            fg_maps.append(fg.detach().cpu().numpy())

            if cam_unflipped is None:
                cam_unflipped = cam[0].detach().cpu().numpy()

        fg_seg = normalize_scoremap(np.mean(np.stack(fg_maps, axis=0), axis=0))
        seg_box = scoremap_to_bbox(fg_seg, threshold=cam_thr)
        seg_iou = bbox_iou(seg_box, gt)

        if cam_unflipped.shape != (H, W):
            fg_cam = (cam_unflipped != 0).astype(np.float32)
            fg_cam = np.asarray(F.interpolate(
                torch.from_numpy(fg_cam)[None, None], size=(H, W), mode='nearest')[0, 0])
        else:
            fg_cam = (cam_unflipped != 0).astype(np.float32)
        cam_box = scoremap_to_bbox(normalize_scoremap(fg_cam), threshold=0.5)
        cam_iou = bbox_iou(cam_box, gt)

        # GT-Known Loc: IoU(predicted box, GT box) >= threshold (class-agnostic)
        seg_box_ok = int(seg_iou >= iou_thr)
        cam_box_ok = int(cam_iou >= iou_thr)

        seg_iou_sum += seg_iou
        cam_iou_sum += cam_iou
        cls_top1_sum += cls_top1
        cls_top5_sum += cls_top5
        gtk_seg_sum += seg_box_ok
        gtk_cam_sum += cam_box_ok
        # Top-1/Top-5 Loc: classification correct AND box correct
        top1_loc_seg_sum += int(cls_top1 and seg_box_ok)
        top5_loc_seg_sum += int(cls_top5 and seg_box_ok)
        top1_loc_cam_sum += int(cls_top1 and cam_box_ok)
        top5_loc_cam_sum += int(cls_top5 and cam_box_ok)
        n_total += 1

        if args.save_vis:
            _save_vis(dataset, name[0], gt, seg_box, cam_box)
            _save_heatmap(dataset, name[0], fg_seg, H, W, 'seg_heatmap')
            _save_cam_heatmap(dataset, name[0], cam_raw_list, gt_cls, H, W)

    n_total = max(1, n_total)
    return {
        # classification accuracy
        'top1_cls': 100.0 * cls_top1_sum / n_total,
        'top5_cls': 100.0 * cls_top5_sum / n_total,
        # GT-Known localization accuracy (box only)
        'gtknown_loc_seg': 100.0 * gtk_seg_sum / n_total,
        'gtknown_loc_cam': 100.0 * gtk_cam_sum / n_total,
        # Top-1 / Top-5 localization accuracy (cls AND box)
        'top1_loc_seg': 100.0 * top1_loc_seg_sum / n_total,
        'top5_loc_seg': 100.0 * top5_loc_seg_sum / n_total,
        'top1_loc_cam': 100.0 * top1_loc_cam_sum / n_total,
        'top5_loc_cam': 100.0 * top5_loc_cam_sum / n_total,
        # mean IoU between predicted and GT boxes
        'seg_mean_iou': 100.0 * seg_iou_sum / n_total,
        'cam_mean_iou': 100.0 * cam_iou_sum / n_total,
        'count': n_total,
    }


def _save_vis(dataset, stem, gt, seg_box, cam_box):
    img_path = os.path.join(dataset.img_dir, stem + '.jpg')
    img = cv2.imread(img_path)
    if img is None:
        return
    gt = gt.astype(int)
    sb = seg_box.astype(int)
    cb = cam_box.astype(int)
    cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)     # GT green
    cv2.rectangle(img, (sb[0], sb[1]), (sb[2], sb[3]), (0, 0, 255), 2)     # seg red
    cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), (255, 0, 0), 2)     # cam blue
    out_path = os.path.join(args.work_dir, 'vis', stem.replace('/', '__') + '.jpg')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def _overlay_heatmap(image, heatmap, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap (H, W) on image (H, W, 3) with alpha blending."""
    heatmap = np.nan_to_num(heatmap, nan=0.0)
    vmin = heatmap.min()
    vmax = heatmap.max()
    if vmax > vmin:
        heatmap_uint8 = np.uint8(255 * (heatmap - vmin) / (vmax - vmin))
    else:
        heatmap_uint8 = np.zeros_like(heatmap, dtype=np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlayed = cv2.addWeighted(image, 0.5, heatmap_color, 0.5, 0)
    return overlayed


def _save_heatmap(dataset, stem, seg_fg, H, W, tag='heatmap'):
    """Save seg foreground heatmap overlaid on the original image."""
    img_path = os.path.join(dataset.img_dir, stem + '.jpg')
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.resize(img, (W, H))

    # seg_fg is already (H, W) from normalize_scoremap at original size
    if seg_fg.shape != (H, W):
        seg_fg = cv2.resize(seg_fg.astype(np.float32), (W, H))
    overlayed = _overlay_heatmap(img, seg_fg)
    out_path = os.path.join(args.work_dir, 'vis', stem.replace('/', '__') + '_' + tag + '.jpg')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlayed)


def _save_cam_heatmap(dataset, stem, cam_raw_list, gt_cls, H, W):
    """Save CAM heatmap for the ground-truth class overlaid on the original image.

    cam_raw_list: list of per-image tensors (K, h_cam, w_cam) where K is
    the number of active classes. We take the slice for gt_cls.
    """
    if cam_raw_list is None or len(cam_raw_list) == 0:
        return
    img_path = os.path.join(dataset.img_dir, stem + '.jpg')
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.resize(img, (W, H))

    # The first element corresponds to the un-flipped image
    cam_tensor = cam_raw_list[0]  # (K, h_cam, w_cam) on CPU
    K = cam_tensor.shape[0]
    # The cam keys correspond to cls indices (1..200) -> search for gt_cls
    # gt_cls is 0-indexed (0..199), whereas cam key indices are 1-indexed
    target_idx = gt_cls + 1  # 1-indexed class id
    # The cam_raw_list tensor uses key ordering; we need to find the index
    # Actually, cam_tensor rows correspond to cam_dict['keys'] ordering.
    # Since we don't pass keys out here, we take the max over the channel dim
    # which is the response map for the GT class if it exists, or max projection.
    cam_np = cam_tensor.max(dim=0)[0].numpy().astype(np.float32)  # fallback: max over classes

    cam_np = cv2.resize(cam_np, (W, H))
    overlayed = _overlay_heatmap(img, cam_np, colormap=cv2.COLORMAP_JET)
    out_path = os.path.join(args.work_dir, 'vis', stem.replace('/', '__') + '_cam_heatmap.jpg')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlayed)


def main(cfg):
    val_dataset = cub.CUBEvalDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    model = RSL(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        sam3_model=cfg.sam3_init.sam3_model,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers=cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device='cuda',
    )

    if args.model_path is None:
        raise ValueError('--model_path must be provided')

    trained_state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    scales = tuple(float(s) for s in args.scales.split(',') if s.strip())
    score = validate(model=model, dataset=val_dataset, cfg=cfg, test_scales=scales)
    torch.cuda.empty_cache()

    iou_thr = float(getattr(cfg.eval, 'iou_threshold', 0.5))
    print('\n================ CUB-200-2011 WSOL results (%s, %d imgs) ================'
          % (args.eval_set, score['count']))
    print('Classification :  Top-1 Cls = %6.2f   Top-5 Cls = %6.2f' % (score['top1_cls'], score['top5_cls']))
    print('---- localization from segmentation foreground map (IoU thr = %.2f) ----' % iou_thr)
    print('  GT-Known Loc = %6.2f   Top-1 Loc = %6.2f   Top-5 Loc = %6.2f   mIoU = %6.2f'
          % (score['gtknown_loc_seg'], score['top1_loc_seg'], score['top5_loc_seg'], score['seg_mean_iou']))
    print('---- localization from CAM pseudo-label (IoU thr = %.2f) ----' % iou_thr)
    print('  GT-Known Loc = %6.2f   Top-1 Loc = %6.2f   Top-5 Loc = %6.2f   mIoU = %6.2f'
          % (score['gtknown_loc_cam'], score['top1_loc_cam'], score['top5_loc_cam'], score['cam_mean_iou']))
    print('=========================================================================')
    print(score)
    return True


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)
    os.makedirs(args.work_dir, exist_ok=True)

    main(cfg=cfg)
