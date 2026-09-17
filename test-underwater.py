"""Multi-scale + flip + dense-CRF evaluation for the AquaOV255 underwater
pipeline.  Mirrors `test_msc_flip_voc.py` / `test-VOC.py` (just with the
underwater dataset / model and a 255-class histogram).
"""

import argparse
import os
import sys

sys.path.append('.')

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import imageio
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import multiprocessing
from tqdm import tqdm

from datasets import underwater
from utils import evaluate
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
from RSL.model_attn_aff_underwater import RSL


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/underwater_attn_reg.yaml', type=str)
parser.add_argument('--work_dir', default='results/underwater', type=str)
parser.add_argument('--eval_set', default='val', type=str)
parser.add_argument('--model_path', default="path/to/rsl_iter_30000.pth", type=str)
parser.add_argument('--scales', default='1,1.5', type=str,
                    help='comma separated test-time scales (besides flip)')


def _remap_raw_mask(raw, ignore_index=255):
    label = raw.astype(np.int32)
    out = np.full(label.shape, ignore_index, dtype=np.int16)
    fg = label != underwater.MASK_IGNORE_RAW
    out[fg] = (label[fg] + 1).astype(np.int16)
    return out


def validate(model, dataset, cfg, test_scales=(1.0,)):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False,
    )
    model.cuda()
    model.eval()

    num_classes = cfg.dataset.num_classes
    _preds_hist = np.zeros((num_classes, num_classes))
    _msc_preds_hist = np.zeros((num_classes, num_classes))
    _cams_hist = np.zeros((num_classes, num_classes))

    _preds, _gts, _msc_preds, cams = [], [], [], []
    num = 0

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
        num += 1
        name, inputs, labels, cls_labels = data
        names = list(name) + list(name)
        cls_labels_cat = torch.cat([cls_labels, cls_labels], dim=0)

        inputs = inputs.cuda()
        labels = labels.cuda()

        _, _, h, w = inputs.shape
        ratio = cfg.clip_init.resize_long / max(h, w)
        _h, _w = int(h * ratio), int(w * ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_clip_cat, segs_dino_cat, cam, _ = model(
            inputs_cat, names, cls_labels=cls_labels_cat, mode='val',
        )
        segs_cat = 0.5 * segs_dino_cat + 0.5 * segs_clip_cat

        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, s_h, s_w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_clip_cat, segs_dino_cat, _, _ = model(
                    inputs_cat, names, cls_labels=cls_labels_cat, mode='val',
                )
                segs_cat = 0.5 * segs_dino_cat + 0.5 * segs_clip_cat
                _segs_cat = F.interpolate(segs_cat, size=(s_h, s_w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0, ...] + _segs_cat[1, ...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))

        if num % 1000 == 0:
            _preds_hist, _ = evaluate.scores(_gts, _preds, _preds_hist, num_classes=num_classes)
            _msc_preds_hist, _ = evaluate.scores(_gts, _msc_preds, _msc_preds_hist, num_classes=num_classes)
            _cams_hist, _ = evaluate.scores(_gts, cams, _cams_hist, num_classes=num_classes)
            _preds, _gts, _msc_preds, cams = [], [], [], []

        np.save(
             os.path.join(args.work_dir, 'logit', name[0] + '.npy'),
             {'segs': segs.detach().cpu().numpy(), 'msc_segs': msc_segs.detach().cpu().numpy()},
         )

    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(cfg):
    print('crf post-processing...')

    num_classes = cfg.dataset.num_classes
    txt_name = os.path.join(cfg.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(cfg.dataset.root_dir, 'images')
    labels_path = os.path.join(cfg.dataset.root_dir, 'masks')

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    def _job(i):
        name = name_list[i]
        logit_name = os.path.join(args.work_dir, 'logit', name + '.npy')

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + '.jpg')
        image = imageio.imread(image_name).astype(np.float32)

        if 'test' in args.eval_set:
            label = image[:, :, 0]
        else:
            label_name = os.path.join(labels_path, name + '.png')
            raw = np.array(Image.open(label_name))
            label = _remap_raw_mask(raw, ignore_index=cfg.dataset.ignore_index)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(
            os.path.join(args.work_dir, 'prediction', name + '.png'),
            np.squeeze(pred).astype(np.uint8),
        )
        imageio.imsave(
            os.path.join(args.work_dir, 'prediction_cmap', name + '.png'),
            encode_cmap(np.squeeze(pred)).astype(np.uint8),
        )
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch='all')(
        [joblib.delayed(_job)(i) for i in range(len(name_list))]
    )

    preds, gts = zip(*results)
    hist = np.zeros((num_classes, num_classes))
    hist, score = evaluate.scores(gts, preds, hist, num_classes=num_classes)
    print(score)
    return True


def main(cfg):
    val_dataset = underwater.UnderwaterSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
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
    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(
        model=model, dataset=val_dataset, cfg=cfg, test_scales=scales,
    )
    torch.cuda.empty_cache()

    num_classes = cfg.dataset.num_classes
    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist, num_classes=num_classes)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist, num_classes=num_classes)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist, num_classes=num_classes)

    print('cams score:')
    print(cam_score)
    print('segs score:')
    print(seg_score)
    print('msc segs score:')
    print(msc_seg_score)

    crf_proc(cfg=cfg)



    
    return True


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)
    os.makedirs(os.path.join(args.work_dir, 'logit'), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'prediction'), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'prediction_cmap'), exist_ok=True)

    main(cfg=cfg)
