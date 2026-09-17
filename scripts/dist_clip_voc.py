import argparse
import copy
import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from RSL.model_attn_aff_voc import RSL
from RSL.dice_loss import DiceLoss
from RSL.prototype_contrastive import PrototypeContrastiveLoss
from RSL.boundary_loss import BoundaryAwareDistillationLoss


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None):
    model.eval()
    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))
    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()
        
        b,c,h,w = inputs.shape
        if (h//cfg.clip_init.clip_flag==0) and (w//cfg.clip_init.clip_flag)==0:
            inputs = inputs
        else:
            new_h = round(h/cfg.clip_init.clip_flag)*cfg.clip_init.clip_flag
            new_w = round(w/cfg.clip_init.clip_flag)*cfg.clip_init.clip_flag
            inputs = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)

        segs_clip,segs_dino, cam, attn_loss = model(inputs, name, 'val')
        segs = 0.5*(segs_dino+segs_clip)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    model.train()
    return seg_score, cam_score


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_uncertainty_confidence(seg_a, seg_b):
    num_classes = seg_a.shape[1]
    max_entropy = np.log(num_classes)

    prob_a = F.softmax(seg_a, dim=1)
    prob_b = F.softmax(seg_b, dim=1)
    log_prob_a = F.log_softmax(seg_a, dim=1)
    log_prob_b = F.log_softmax(seg_b, dim=1)

    # branch-wise entropy, normalized to [0, 1]
    entropy_a = -(prob_a * log_prob_a).sum(dim=1) / max_entropy
    entropy_b = -(prob_b * log_prob_b).sum(dim=1) / max_entropy
    entropy_uncertainty = 0.5 * (entropy_a + entropy_b)

    # symmetric KL as branch disagreement term
    kl_ab = (prob_a * (log_prob_a - log_prob_b)).sum(dim=1)
    kl_ba = (prob_b * (log_prob_b - log_prob_a)).sum(dim=1)
    skl = 0.5 * (kl_ab + kl_ba)
    disagreement = 1.0 - torch.exp(-skl)  # map to [0, 1)

    uncertainty = 0.5 * entropy_uncertainty + 0.5 * disagreement
    confidence = (1.0 - uncertainty).clamp(min=0.0, max=1.0)
    return confidence


def get_uncertainty_weighted_ce_loss(pred, label, confidence, ignore_index=255):
    ce_per_pixel = F.cross_entropy(
        pred, label.long(), ignore_index=ignore_index, reduction='none'
    )
    valid_mask = (label != ignore_index).float()
    if valid_mask.sum() == 0:
        return pred.sum() * 0.0

    weighted = ce_per_pixel * confidence.detach() * valid_mask
    return weighted.sum() / valid_mask.sum()


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def update_ema_teacher(student, teacher, ema_decay):
    with torch.no_grad():
        student_state = student.state_dict()
        teacher_state = teacher.state_dict()
        for k, v in teacher_state.items():
            s = student_state[k]
            if v.dtype.is_floating_point:
                v.mul_(ema_decay).add_(s, alpha=1.0 - ema_decay)
            else:
                v.copy_(s)


def build_ema_replace_mask(conf_map, valid_mask, replace_ratio, conf_thresh):
    if replace_ratio <= 0.0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    teacher_mask = (conf_map > conf_thresh) & valid_mask
    if replace_ratio < 1.0:
        rand_mask = torch.rand_like(conf_map) < replace_ratio
        teacher_mask = teacher_mask & rand_mask
    return teacher_mask


def get_ema_replace_ratio(n_iter, start_iter, ramp_iters, max_ratio):
    if n_iter < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return max_ratio
    progress = float(n_iter - start_iter + 1) / float(ramp_iters)
    return max_ratio * min(1.0, max(0.0, progress))



def train(cfg):

    num_workers = 10
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='train',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)


    model = RSL(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        sam3_model=cfg.sam3_init.sam3_model,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim = cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers = cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device='cuda'
    )
    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model.cuda()
    teacher_model = copy.deepcopy(model).cuda()
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    mask_size = int(cfg.dataset.crop_size // cfg.clip_init.clip_flag)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    criterion_dice = DiceLoss().cuda()

    proto_cfg = getattr(cfg, 'prototype', None)
    proto_momentum = float(getattr(proto_cfg, 'momentum', 0.999)) if proto_cfg else 0.999
    proto_temperature = float(getattr(proto_cfg, 'temperature', 0.1)) if proto_cfg else 0.1
    proto_conf_thresh = float(getattr(proto_cfg, 'confidence_thresh', 0.7)) if proto_cfg else 0.7
    proto_loss_weight = float(getattr(proto_cfg, 'loss_weight', 0.1)) if proto_cfg else 0.1
    criterion_proto = PrototypeContrastiveLoss(
        num_classes=cfg.dataset.num_classes,
        embedding_dim=cfg.clip_init.embedding_dim,
        momentum=proto_momentum,
        temperature=proto_temperature,
        confidence_thresh=proto_conf_thresh,
    ).cuda()

    bdry_cfg = getattr(cfg, 'boundary', None)
    bdry_temperature = float(getattr(bdry_cfg, 'temperature', 2.0)) if bdry_cfg else 2.0
    bdry_dilation = int(getattr(bdry_cfg, 'dilation', 2)) if bdry_cfg else 2
    bdry_kl_weight = float(getattr(bdry_cfg, 'kl_weight', 1.0)) if bdry_cfg else 1.0
    bdry_fidelity_weight = float(getattr(bdry_cfg, 'fidelity_weight', 1.0)) if bdry_cfg else 1.0
    bdry_loss_weight = float(getattr(bdry_cfg, 'loss_weight', 0.1)) if bdry_cfg else 0.1
    criterion_boundary = BoundaryAwareDistillationLoss(
        temperature=bdry_temperature,
        boundary_dilation=bdry_dilation,
        kl_weight=bdry_kl_weight,
        fidelity_weight=bdry_fidelity_weight,
    ).cuda()

    ema_decay = float(getattr(cfg.train, "ema_decay", 0.999))
    ema_replace_start = int(getattr(cfg.train, "ema_replace_start", 2000))
    ema_ramp_iters = int(getattr(cfg.train, "ema_ramp_iters", 10000))
    ema_conf_thresh = float(getattr(cfg.train, "ema_conf_thresh", 0.7))
    ema_max_replace_ratio = float(getattr(cfg.train, "ema_max_replace_ratio", 0.9))


    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs_clip, segs_dino, cam, attn_pred, fts, dino_fts = model(inputs.cuda(), img_name)

        pseudo_label = cam

        segs= 0.5*segs_clip+0.5*segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()

        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index, clip_flag=cfg.clip_init.clip_flag)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        with torch.no_grad():
            tea_clip, tea_dino = teacher_model.forward_seg(inputs.cuda())
            tea_seg = 0.5 * (tea_clip + tea_dino)
            tea_seg = F.interpolate(tea_seg, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            tea_prob = F.softmax(tea_seg, dim=1)
            tea_conf, tea_label = torch.max(tea_prob, dim=1)

            replace_ratio = get_ema_replace_ratio(
                n_iter=n_iter,
                start_iter=ema_replace_start,
                ramp_iters=ema_ramp_iters,
                max_ratio=ema_max_replace_ratio,
            )
            valid_mask = pseudo_label != cfg.dataset.ignore_index
            teacher_mask = build_ema_replace_mask(
                conf_map=tea_conf,
                valid_mask=valid_mask,
                replace_ratio=replace_ratio,
                conf_thresh=ema_conf_thresh,
            )
            pseudo_label = torch.where(teacher_mask, tea_label, pseudo_label)
            teacher_replace_rate = teacher_mask.float().mean()

        confidence = get_uncertainty_confidence(segs_clip, segs_dino)
        pseudo_label_for_ce = pseudo_label.long().clone()
        pseudo_label_for_dice = pseudo_label.long().clone()

        seg_loss = get_uncertainty_weighted_ce_loss(
            segs, pseudo_label_for_ce, confidence, ignore_index=cfg.dataset.ignore_index
        )
        seg_loss2 = criterion_dice(segs, pseudo_label_for_dice)

        seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        proto_loss, proto_sam3, proto_dino, proto_align = criterion_proto(
            fts, dino_fts, pseudo_label, confidence
        )

        bdry_loss, bdry_kl, bdry_fid = criterion_boundary(
            segs_clip, segs_dino, inputs.cuda(),
            pseudo_label=pseudo_label,
            ignore_index=cfg.dataset.ignore_index,
        )

        loss = (1 * seg_loss + 0.1 * attn_loss
                + 0.1 * (seg_clip_loss2 + seg_dino_loss2)
                + 1 * seg_loss2
                + 0.1 * proto_loss_weight * proto_loss
                + bdry_loss_weight * bdry_loss)

        avg_meter.add({
            'seg_loss': seg_loss.item(),
            'attn_loss': attn_loss.item(),
            'proto_loss': proto_loss.item(),
            'proto_align': proto_align.item(),
            'bdry_loss': bdry_loss.item(),
            'bdry_kl': bdry_kl.item(),
            'bdry_fid': bdry_fid.item(),
            'conf_mean': confidence.mean().item(),
            'ema_replace_rate': teacher_replace_rate.item(),
            'ema_replace_ratio': replace_ratio,
            'tea_conf_mean': tea_conf.mean().item(),
        })

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_teacher(model, teacher_model, ema_decay=ema_decay)

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            logging.info(
                "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, "
                "attn_loss: %.4f, proto_loss: %.4f, proto_align: %.4f, "
                "bdry_loss: %.4f, bdry_kl: %.4f, bdry_fid: %.4f, "
                "conf_mean: %.4f, ema_replace_rate: %.4f, "
                "ema_target_ratio: %.4f, tea_conf_mean: %.4f, pseudo_seg_mAcc: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr, avg_meter.pop('seg_loss'),
                    avg_meter.pop('attn_loss'),
                    avg_meter.pop('proto_loss'), avg_meter.pop('proto_align'),
                    avg_meter.pop('bdry_loss'), avg_meter.pop('bdry_kl'),
                    avg_meter.pop('bdry_fid'),
                    avg_meter.pop('conf_mean'),
                    avg_meter.pop('ema_replace_rate'), avg_meter.pop('ema_replace_ratio'),
                    avg_meter.pop('tea_conf_mean'), seg_mAcc
                )
            )

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item(), "proto_loss": proto_loss.item(), "proto_align": proto_align.item(), "bdry_loss": bdry_loss.item(), "bdry_kl": bdry_kl.item(), "bdry_fid": bdry_fid.item()}, global_step=n_iter)
            writer.add_scalars(
                'train/ema',
                {
                    "replace_rate": teacher_replace_rate.item(),
                    "target_ratio": replace_ratio,
                    "teacher_conf_mean": tea_conf.mean().item(),
                },
                global_step=n_iter
            )


        if (n_iter +1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "rsl_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            # if (n_iter + 1) > 26000:
            if (n_iter + 1) > 500:
                torch.save(model.state_dict(), ckpt_name)
            seg_score, cam_score = validate(model=model, data_loader=val_loader, cfg=cfg)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)


