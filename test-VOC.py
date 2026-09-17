# import argparse
# import os
# import sys
# sys.path.append(".")
# import cv2
# from utils.dcrf import DenseCRF
# from utils.imutils import encode_cmap
# import numpy as np
# import torch
# import torch.nn.functional as F
# from omegaconf import OmegaConf
# from torch import multiprocessing
# from tqdm import tqdm
# import joblib
# from datasets import voc
# from utils import evaluate
# from RSL.model_attn_aff_voc import RSL
# import imageio
# from PIL import Image as PILImage, ImageDraw, ImageFont

# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from sklearn.manifold import TSNE
#     matplotlib.rcParams['font.family'] = 'serif'
#     matplotlib.rcParams['font.serif'] = [
#         'Times New Roman',
#         'Times',
#         'Nimbus Roman',
#         'Liberation Serif',
#         'DejaVu Serif',
#     ]
#     matplotlib.rcParams['axes.unicode_minus'] = False
#     _TSNE_AVAILABLE = True
# except Exception:
#     _TSNE_AVAILABLE = False

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
# NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# parser = argparse.ArgumentParser()
# parser.add_argument("--config",
#                     default='configs/voc_attn_reg.yaml',
#                     type=str,
#                     help="config")
# parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
# parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
# parser.add_argument("--eval_set", default="val", type=str, help="eval_set") #val
# parser.add_argument("--model_path", default="path/to/rsl_iter_30000.pth", type=str, help="model_path")
# # parser.add_argument("--model_path", default="path/to/rsl_iter_1000.pth", type=str, help="model_path")
# # parser.add_argument("--model_path", default="path/to/rsl_iter_2000.pth", type=str, help="model_path")
# parser.add_argument("--save_vis", default=True, type=lambda x: str(x).lower() in ["true", "1", "yes"], help="save visualization results")
# parser.add_argument("--vis_max_images", default=-1, type=int, help="max images for visualization, -1 means all")
# parser.add_argument("--vis_topk_channels", default=8, type=int, help="top-k feature channels to visualize per backbone")
# parser.add_argument("--tsne_points_per_class_per_image", default=8, type=int, help="t-SNE sampled points per class per image per model")
# parser.add_argument("--tsne_max_points_per_model", default=6000, type=int, help="max total t-SNE points per model on full dataset")


# _TIMES_FONT_CANDIDATES = [
#     "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
#     "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
#     "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
#     "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
#     "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
# ]


# def _get_times_font(font_size=16):
#     for fp in _TIMES_FONT_CANDIDATES:
#         if os.path.exists(fp):
#             try:
#                 return ImageFont.truetype(fp, size=font_size)
#             except Exception:
#                 continue
#     return ImageFont.load_default()


# def draw_text_times(img_rgb, text, org_xy=(5, 18), font_size=16, color=(255, 255, 255)):
#     """Draw text using Times New Roman family (or serif fallback) on RGB numpy image."""
#     pil_img = PILImage.fromarray(img_rgb)
#     draw = ImageDraw.Draw(pil_img)
#     font = _get_times_font(font_size=font_size)
#     draw.text(org_xy, text, font=font, fill=color)
#     return np.asarray(pil_img)


# def denormalize_to_uint8(image_tensor):
#     img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
#     img = img * NORM_STD + NORM_MEAN
#     img = np.clip(img, 0, 255).astype(np.uint8)
#     return img


# def normalize_map(x):
#     x = x.astype(np.float32)
#     x_min, x_max = float(x.min()), float(x.max())
#     if x_max - x_min < 1e-6:
#         return np.zeros_like(x, dtype=np.float32)
#     return (x - x_min) / (x_max - x_min)


# def make_heatmap_rgb(score_map):
#     score_map = normalize_map(score_map)
#     hm_uint8 = (score_map * 255).astype(np.uint8)
#     hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
#     hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
#     return hm


# def save_heatmap(score_map, save_path):
#     heatmap_rgb = make_heatmap_rgb(score_map)
#     imageio.imsave(save_path, heatmap_rgb.astype(np.uint8))


# def save_overlay(image_rgb, score_map, save_path, alpha=0.45):
#     heatmap_rgb = make_heatmap_rgb(score_map)
#     if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
#         heatmap_rgb = cv2.resize(heatmap_rgb, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
#     overlay = (image_rgb.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha)
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#     imageio.imsave(save_path, overlay)


# def save_feature_channel_grid(feature_map, save_path, topk=8, tile_hw=128):
#     if feature_map is None:
#         return
#     feat = feature_map.detach().cpu().float()
#     if feat.dim() == 4:
#         feat = feat[0]
#     c, h, w = feat.shape
#     topk = min(int(topk), int(c))
#     if topk <= 0:
#         return

#     ch_score = feat.abs().mean(dim=(1, 2))
#     ch_idx = torch.topk(ch_score, k=topk, largest=True).indices.tolist()

#     tiles = []
#     for ci in ch_idx:
#         ch_map = feat[ci].numpy()
#         ch_map = normalize_map(ch_map)
#         tile = make_heatmap_rgb(ch_map)
#         tile = cv2.resize(tile, (tile_hw, tile_hw), interpolation=cv2.INTER_LINEAR)
#         tile = draw_text_times(tile, f"ch={ci}", org_xy=(5, 6), font_size=16, color=(255, 255, 255))
#         tiles.append(tile)

#     cols = 4
#     rows = int(np.ceil(len(tiles) / cols))
#     canvas = np.zeros((rows * tile_hw, cols * tile_hw, 3), dtype=np.uint8)
#     for i, tile in enumerate(tiles):
#         r, c_idx = divmod(i, cols)
#         canvas[r*tile_hw:(r+1)*tile_hw, c_idx*tile_hw:(c_idx+1)*tile_hw] = tile

#     imageio.imsave(save_path, canvas)


# def save_tsne_plot(feature_dict, save_path, max_points=400):
#     if not _TSNE_AVAILABLE:
#         return

#     feats_all, tags_all = [], []
#     feat_dims = []
#     for tag, feat in feature_dict.items():
#         if feat is None:
#             continue
#         f = feat.detach().cpu().float()
#         if f.dim() == 4:
#             f = f[0]
#         c, h, w = f.shape
#         f = f.reshape(c, h * w).transpose(0, 1).numpy()
#         n = f.shape[0]
#         if n == 0:
#             continue
#         sample_n = min(max_points, n)
#         idx = np.random.choice(n, size=sample_n, replace=False)
#         f_sample = f[idx]
#         feats_all.append(f_sample)
#         feat_dims.append(f_sample.shape[1])
#         tags_all.extend([tag] * sample_n)

#     if len(feats_all) == 0:
#         return

#     # Different backbones may have different channel dimensions.
#     # Pad to the same width before concatenation.
#     max_dim = int(max(feat_dims))
#     aligned_feats = []
#     for f in feats_all:
#         cur_dim = f.shape[1]
#         if cur_dim < max_dim:
#             f = np.pad(f, ((0, 0), (0, max_dim - cur_dim)), mode='constant')
#         aligned_feats.append(f)

#     x = np.concatenate(aligned_feats, axis=0)
#     if x.shape[0] < 10:
#         return

#     perplexity = max(5, min(30, x.shape[0] // 10))
#     tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=0)
#     emb = tsne.fit_transform(x)

#     plt.figure(figsize=(7, 6))
#     tags_all = np.array(tags_all)
#     for tag in np.unique(tags_all):
#         mask = tags_all == tag
#         plt.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.65, label=tag)
#     plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'family': 'Times New Roman'})
#     plt.title("Feature t-SNE clustering", fontfamily='Times New Roman')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200)
#     plt.close()


# def _sample_features_by_class(feature_map, label_map, points_per_class=8):
#     """
#     feature_map: (C, H, W) tensor
#     label_map:   (H, W) tensor, VOC labels in [0..20], ignore=255
#     returns:
#         sampled_feats: (N, C) np.ndarray
#         sampled_labels: (N,) np.ndarray
#     """
#     feat = feature_map.detach().cpu().float()
#     if feat.dim() == 4:
#         feat = feat[0]
#     c, h, w = feat.shape

#     lab = label_map.detach().cpu().long()
#     if lab.dim() == 3:
#         lab = lab[0]
#     if lab.shape[0] != h or lab.shape[1] != w:
#         lab_rs = F.interpolate(lab.unsqueeze(0).unsqueeze(0).float(), size=(h, w), mode='nearest')
#         lab = lab_rs.squeeze(0).squeeze(0).long()

#     feat_flat = feat.reshape(c, h * w).transpose(0, 1).numpy()  # (HW, C)
#     lab_flat = lab.reshape(-1).numpy()  # (HW,)

#     keep_mask = lab_flat != 255
#     feat_flat = feat_flat[keep_mask]
#     lab_flat = lab_flat[keep_mask]
#     if feat_flat.shape[0] == 0:
#         return None, None

#     sampled_feat_list, sampled_lab_list = [], []
#     for cls_id in np.unique(lab_flat):
#         cls_idx = np.where(lab_flat == cls_id)[0]
#         if cls_idx.size == 0:
#             continue
#         take_n = min(int(points_per_class), int(cls_idx.size))
#         chosen = np.random.choice(cls_idx, size=take_n, replace=False)
#         sampled_feat_list.append(feat_flat[chosen])
#         sampled_lab_list.append(np.full((take_n,), cls_id, dtype=np.int64))

#     if len(sampled_feat_list) == 0:
#         return None, None

#     return np.concatenate(sampled_feat_list, axis=0), np.concatenate(sampled_lab_list, axis=0)


# def save_dataset_tsne(tsne_bank, save_dir, max_points_per_model=6000):
#     if not _TSNE_AVAILABLE:
#         return

#     for model_tag, data in tsne_bank.items():
#         if len(data['feats']) == 0:
#             continue

#         x = np.concatenate(data['feats'], axis=0)
#         y = np.concatenate(data['labels'], axis=0)
#         if x.shape[0] < 10:
#             continue

#         if x.shape[0] > int(max_points_per_model):
#             idx = np.random.choice(x.shape[0], size=int(max_points_per_model), replace=False)
#             x = x[idx]
#             y = y[idx]

#         perplexity = max(5, min(40, x.shape[0] // 20))
#         tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=0)
#         emb = tsne.fit_transform(x)

#         plt.figure(figsize=(8, 7))
#         uniq_cls = np.unique(y)
#         cmap = plt.cm.get_cmap('tab20', 21)
#         for cls_id in uniq_cls:
#             m = y == cls_id
#             color = cmap(int(cls_id) % 21)
#             plt.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.65, color=color, label=str(int(cls_id)))

#         plt.legend(title='VOC cls id', fontsize=8, ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'family': 'Times New Roman'})
#         plt.title(f"{model_tag} dataset-level t-SNE (all classes)", fontfamily='Times New Roman')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f"dataset_tsne_{model_tag.lower()}.png"), dpi=220)
#         plt.close()


# def validate(model, dataset, test_scales=None):

#     _preds, _gts, _msc_preds, cams = [], [], [], []
    
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
#     model.cuda()
#     model.eval()

#     num = 0

#     _preds_hist = np.zeros((21, 21))
#     _msc_preds_hist = np.zeros((21, 21))
#     _cams_hist = np.zeros((21, 21))

#     tsne_bank = {
#         'SAM3': {'feats': [], 'labels': []},
#         'DINO': {'feats': [], 'labels': []},
#         'CLIP': {'feats': [], 'labels': []},
#     }

#     for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
#         num+=1

#         name, inputs, labels, cls_labels = data
#         names = name+name

#         inputs = inputs.cuda()
#         labels = labels.cuda()
#         inputs_for_vis = inputs.clone()

#         _, _, h, w = inputs.shape
#         ratio = cfg.clip_init.resize_long / max(h,w)
#         _h, _w = int(h*ratio), int(w*ratio)
#         inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

#         segs_list = []
#         inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
#         if args.save_vis:
#             segs_clip_cat, segs_dino_cat, cam, attn_loss, vis_payload = model(inputs_cat, names, mode='val', return_vis=True)
#         else:
#             segs_clip_cat, segs_dino_cat, cam, attn_loss = model(inputs_cat, names, mode='val')
#             vis_payload = None

#         segs_cat = 0.5 * segs_dino_cat + 0.5*segs_clip_cat
        
#         cam = cam[0].unsqueeze(0)
#         segs = segs_cat[0].unsqueeze(0)

#         _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
#         segs_list.append(_segs)

#         _, _, s_h, s_w = segs_cat.shape

#         for s in test_scales:
#             if s != 1.0:
#                 _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
#                 inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

#                 segs_clip_cat, segs_dino_cat, cam_cat, attn_loss = model(inputs_cat, names, mode='val')

#                 segs_cat = 0.5* segs_dino_cat + 0.5*segs_clip_cat

#                 _segs_cat = F.interpolate(segs_cat, size=(s_h, s_w), mode='bilinear', align_corners=False)
#                 _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
#                 segs_list.append(_segs)

#         msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)
        
#         resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
#         seg_preds = torch.argmax(resized_segs, dim=1)
#         # print('seg_shape', seg_preds.shape, 'labels', labels.shape, 'cam', cam.shape)

#         resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
#         msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

#         cams += list(cam.cpu().numpy().astype(np.int16))
#         _preds += list(seg_preds.cpu().numpy().astype(np.int16))
#         _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
#         _gts += list(labels.cpu().numpy().astype(np.int16))

#         if args.save_vis and (args.vis_max_images < 0 or idx < args.vis_max_images):
#             img_name = name[0]
#             image_rgb = denormalize_to_uint8(inputs_for_vis[0])

#             # Segmentation confidence heatmap
#             seg_conf = torch.softmax(resized_msc_segs, dim=1)[0].max(dim=0)[0].detach().cpu().numpy()
#             save_heatmap(seg_conf, os.path.join(args.work_dir, "vis", "heatmap", img_name + ".png"))

#             # CAM score map and overlay on original image
#             if vis_payload is not None and vis_payload.get('cam_score_maps', None) is not None:
#                 cam_score_map = vis_payload['cam_score_maps'][0].numpy()
#             else:
#                 cam_score_map = cam[0].detach().cpu().numpy().astype(np.float32)
#             if cam_score_map.shape[:2] != image_rgb.shape[:2]:
#                 cam_score_map = cv2.resize(cam_score_map, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

#             save_heatmap(cam_score_map, os.path.join(args.work_dir, "vis", "cam_heatmap", img_name + ".png"))
#             save_overlay(image_rgb, cam_score_map, os.path.join(args.work_dir, "vis", "cam_overlay", img_name + ".png"))

#             # CAM pseudo label and segmentation outputs
#             cam_label = cam[0].detach().cpu().numpy().astype(np.uint8)
#             imageio.imsave(os.path.join(args.work_dir, "vis", "cam_label", img_name + ".png"), cam_label)
#             imageio.imsave(os.path.join(args.work_dir, "vis", "cam_label_cmap", img_name + ".png"), encode_cmap(cam_label).astype(np.uint8))

#             seg_pred_np = seg_preds[0].detach().cpu().numpy().astype(np.uint8)
#             msc_seg_pred_np = msc_seg_preds[0].detach().cpu().numpy().astype(np.uint8)
#             imageio.imsave(os.path.join(args.work_dir, "vis", "seg", img_name + ".png"), seg_pred_np)
#             imageio.imsave(os.path.join(args.work_dir, "vis", "seg_cmap", img_name + ".png"), encode_cmap(seg_pred_np).astype(np.uint8))
#             imageio.imsave(os.path.join(args.work_dir, "vis", "msc_seg", img_name + ".png"), msc_seg_pred_np)
#             imageio.imsave(os.path.join(args.work_dir, "vis", "msc_seg_cmap", img_name + ".png"), encode_cmap(msc_seg_pred_np).astype(np.uint8))

#             # Different foundation model feature channel heatmaps
#             if vis_payload is not None:
#                 sam3_fts = vis_payload.get('sam3_fts', None)
#                 dino_fts = vis_payload.get('dino_fts', None)
#                 clip_fts = vis_payload.get('clip_fts', None)

#                 save_feature_channel_grid(
#                     sam3_fts[0] if sam3_fts is not None else None,
#                     os.path.join(args.work_dir, "vis", "feature_channels", "sam3", img_name + ".png"),
#                     topk=args.vis_topk_channels,
#                 )
#                 save_feature_channel_grid(
#                     dino_fts[0] if dino_fts is not None else None,
#                     os.path.join(args.work_dir, "vis", "feature_channels", "dino", img_name + ".png"),
#                     topk=args.vis_topk_channels,
#                 )
#                 save_feature_channel_grid(
#                     clip_fts[0] if clip_fts is not None else None,
#                     os.path.join(args.work_dir, "vis", "feature_channels", "clip", img_name + ".png"),
#                     topk=args.vis_topk_channels,
#                 )

#                 # Collect dataset-level t-SNE samples (all classes across all test images)
#                 for model_tag, model_feat in [('SAM3', sam3_fts), ('DINO', dino_fts), ('CLIP', clip_fts)]:
#                     if model_feat is None:
#                         continue
#                     sampled_feats, sampled_labels = _sample_features_by_class(
#                         model_feat[0],
#                         labels[0],
#                         points_per_class=args.tsne_points_per_class_per_image,
#                     )
#                     if sampled_feats is None:
#                         continue
#                     tsne_bank[model_tag]['feats'].append(sampled_feats)
#                     tsne_bank[model_tag]['labels'].append(sampled_labels)


#         if num % 1000 == 0:
#             _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
#             _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
#             _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
#             _preds, _gts, _msc_preds, cams = [], [], [], []


#         np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})

#     if args.save_vis:
#         save_dataset_tsne(
#             tsne_bank=tsne_bank,
#             save_dir=os.path.join(args.work_dir, "vis", "tsne"),
#             max_points_per_model=args.tsne_max_points_per_model,
#         )
            
#     return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


# def crf_proc(config):
#     print("crf post-processing...")

#     txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
#     with open(txt_name) as f:
#         name_list = [x for x in f.read().split('\n') if x]

#     images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
#     labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

#     post_processor = DenseCRF(
#         iter_max=10,    # 10
#         pos_xy_std=3,   # 3
#         pos_w=3,        # 3
#         bi_xy_std=64,  # 64
#         bi_rgb_std=5,   # 5
#         bi_w=4,         # 4
#     )

#     def _job(i):

#         name = name_list[i]
#         logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

#         logit = np.load(logit_name, allow_pickle=True).item()
#         logit = logit['msc_segs']

#         image_name = os.path.join(images_path, name + ".jpg")
#         image = imageio.imread(image_name).astype(np.float32)
#         label_name = os.path.join(labels_path, name + ".png")
#         if "test" in args.eval_set:
#             label = image[:,:,0]
#         else:
#             label = imageio.imread(label_name)

#         H, W, _ = image.shape
#         logit = torch.FloatTensor(logit)#[None, ...]
#         logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
#         prob = F.softmax(logit, dim=1)[0].numpy()

#         image = image.astype(np.uint8)
#         prob = post_processor(image, prob)
#         pred = np.argmax(prob, axis=0)

#         imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
#         imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
#         return pred, label

#     n_jobs = int(multiprocessing.cpu_count() * 0.8)
#     results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

#     preds, gts = zip(*results)
#     hist = np.zeros((21, 21))
#     hist, score = evaluate.scores(gts, preds, hist, 21)

#     print(score)
    
#     return True

# def main(cfg):
    
#     val_dataset = voc.VOC12SegDataset(
#         root_dir=cfg.dataset.root_dir,
#         name_list_dir=cfg.dataset.name_list_dir,
#         split=args.eval_set,
#         stage='val',
#         aug=False,
#         ignore_index=cfg.dataset.ignore_index,
#         num_classes=cfg.dataset.num_classes,
#     )

#     model = RSL(num_classes=cfg.dataset.num_classes,
#                      clip_model=cfg.clip_init.clip_pretrain_path,
#                      dino_model=cfg.dino_init.dino_model,
#                      dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
#                      decoder_layers=cfg.dino_init.decoder_layer,
#                      embedding_dim=cfg.clip_init.embedding_dim,
#                      in_channels=cfg.clip_init.in_channels,
#                      dataset_root_path=cfg.dataset.root_dir,
#                      clip_flag=cfg.clip_init.clip_flag,
#                      device='cuda')
    
#     trained_state_dict = torch.load(args.model_path, map_location="cpu")

#     model.load_state_dict(state_dict=trained_state_dict, strict=False)
#     model.eval()

#     gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(model=model, dataset=val_dataset, test_scales=[1, 1.5]) #[1, 0.75]
#     torch.cuda.empty_cache()

#     preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist)
#     msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist)
#     cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist)

#     print("cams score:")
#     print(cam_score)
#     print("segs score:")
#     print(seg_score)
#     print("msc segs score:")
#     print(msc_seg_score)

#     crf_proc(config=cfg)

#     return True


# if __name__ == "__main__":

#     args = parser.parse_args()
#     cfg = OmegaConf.load(args.config)
#     print(cfg)
#     print(args)

#     args.work_dir = os.path.join(args.work_dir, args.eval_set)

#     os.makedirs(args.work_dir + "/logit", exist_ok=True)
#     os.makedirs(args.work_dir + "/prediction", exist_ok=True)
#     os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

#     if args.save_vis:
#         os.makedirs(args.work_dir + "/vis/heatmap", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/cam_heatmap", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/cam_overlay", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/cam_label", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/cam_label_cmap", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/seg", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/seg_cmap", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/msc_seg", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/msc_seg_cmap", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/feature_channels/sam3", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/feature_channels/dino", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/feature_channels/clip", exist_ok=True)
#         os.makedirs(args.work_dir + "/vis/tsne", exist_ok=True)

#     main(cfg=cfg)























































import argparse
import os
import sys
sys.path.append(".")
import cv2
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from RSL.model_attn_aff_voc_dinov2 import RSL
import imageio
from PIL import Image as PILImage, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = [
        'Times New Roman',
        'Times',
        'Nimbus Roman',
        'Liberation Serif',
        'DejaVu Serif',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    _TSNE_AVAILABLE = True
except Exception:
    _TSNE_AVAILABLE = False

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc_attn_reg_dinov2.yaml',
                    type=str,
                    help="config")
parser.add_argument("--work_dir", default="results-voc-dinov2", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set") #val
parser.add_argument("--model_path", default="path/to/rsl_iter_28000.pth", type=str, help="model_path")
# parser.add_argument("--model_path", default="path/to/rsl_iter_1000.pth", type=str, help="model_path")
# parser.add_argument("--model_path", default="path/to/rsl_iter_2000.pth", type=str, help="model_path")
parser.add_argument("--save_vis", default=True, type=lambda x: str(x).lower() in ["true", "1", "yes"], help="save visualization results")
parser.add_argument("--vis_max_images", default=-1, type=int, help="max images for visualization, -1 means all")
parser.add_argument("--vis_topk_channels", default=8, type=int, help="top-k feature channels to visualize per backbone")
parser.add_argument("--tsne_points_per_class_per_image", default=8, type=int, help="t-SNE sampled points per class per image per model")
parser.add_argument("--tsne_max_points_per_model", default=6000, type=int, help="max total t-SNE points per model on full dataset")
parser.add_argument("--save_grad_bar", default=True, type=lambda x: str(x).lower() in ["true", "1", "yes"], help="save gradient contribution bar chart")
parser.add_argument("--grad_bar_max_images", default=-1, type=int, help="max images used for gradient bar chart, -1 means all")


_TIMES_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]


def _get_times_font(font_size=16):
    for fp in _TIMES_FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text_times(img_rgb, text, org_xy=(5, 18), font_size=16, color=(255, 255, 255)):
    """Draw text using Times New Roman family (or serif fallback) on RGB numpy image."""
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_times_font(font_size=font_size)
    draw.text(org_xy, text, font=font, fill=color)
    return np.asarray(pil_img)


def denormalize_to_uint8(image_tensor):
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * NORM_STD + NORM_MEAN
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def normalize_map(x):
    x = x.astype(np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def make_heatmap_rgb(score_map):
    score_map = normalize_map(score_map)
    hm_uint8 = (score_map * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return hm


def save_heatmap(score_map, save_path):
    heatmap_rgb = make_heatmap_rgb(score_map)
    imageio.imsave(save_path, heatmap_rgb.astype(np.uint8))


def save_overlay(image_rgb, score_map, save_path, alpha=0.45):
    heatmap_rgb = make_heatmap_rgb(score_map)
    if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = (image_rgb.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    imageio.imsave(save_path, overlay)


def save_feature_channel_grid(feature_map, save_path, topk=8, tile_hw=128):
    if feature_map is None:
        return
    feat = feature_map.detach().cpu().float()
    if feat.dim() == 4:
        feat = feat[0]
    c, h, w = feat.shape
    topk = min(int(topk), int(c))
    if topk <= 0:
        return

    ch_score = feat.abs().mean(dim=(1, 2))
    ch_idx = torch.topk(ch_score, k=topk, largest=True).indices.tolist()

    tiles = []
    for ci in ch_idx:
        ch_map = feat[ci].numpy()
        ch_map = normalize_map(ch_map)
        tile = make_heatmap_rgb(ch_map)
        tile = cv2.resize(tile, (tile_hw, tile_hw), interpolation=cv2.INTER_LINEAR)
        tile = draw_text_times(tile, f"ch={ci}", org_xy=(5, 6), font_size=16, color=(255, 255, 255))
        tiles.append(tile)

    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * tile_hw, cols * tile_hw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c_idx = divmod(i, cols)
        canvas[r*tile_hw:(r+1)*tile_hw, c_idx*tile_hw:(c_idx+1)*tile_hw] = tile

    imageio.imsave(save_path, canvas)


def save_tsne_plot(feature_dict, save_path, max_points=400):
    if not _TSNE_AVAILABLE:
        return

    feats_all, tags_all = [], []
    feat_dims = []
    for tag, feat in feature_dict.items():
        if feat is None:
            continue
        f = feat.detach().cpu().float()
        if f.dim() == 4:
            f = f[0]
        c, h, w = f.shape
        f = f.reshape(c, h * w).transpose(0, 1).numpy()
        n = f.shape[0]
        if n == 0:
            continue
        sample_n = min(max_points, n)
        idx = np.random.choice(n, size=sample_n, replace=False)
        f_sample = f[idx]
        feats_all.append(f_sample)
        feat_dims.append(f_sample.shape[1])
        tags_all.extend([tag] * sample_n)

    if len(feats_all) == 0:
        return

    # Different backbones may have different channel dimensions.
    # Pad to the same width before concatenation.
    max_dim = int(max(feat_dims))
    aligned_feats = []
    for f in feats_all:
        cur_dim = f.shape[1]
        if cur_dim < max_dim:
            f = np.pad(f, ((0, 0), (0, max_dim - cur_dim)), mode='constant')
        aligned_feats.append(f)

    x = np.concatenate(aligned_feats, axis=0)
    if x.shape[0] < 10:
        return

    perplexity = max(5, min(30, x.shape[0] // 10))
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=0)
    emb = tsne.fit_transform(x)

    plt.figure(figsize=(7, 6))
    tags_all = np.array(tags_all)
    for tag in np.unique(tags_all):
        mask = tags_all == tag
        plt.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.65, label=tag)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'family': 'Times New Roman'})
    plt.title("Feature t-SNE clustering", fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _sample_features_by_class(feature_map, label_map, points_per_class=8):
    """
    feature_map: (C, H, W) tensor
    label_map:   (H, W) tensor, VOC labels in [0..20], ignore=255
    returns:
        sampled_feats: (N, C) np.ndarray
        sampled_labels: (N,) np.ndarray
    """
    feat = feature_map.detach().cpu().float()
    if feat.dim() == 4:
        feat = feat[0]
    c, h, w = feat.shape

    lab = label_map.detach().cpu().long()
    if lab.dim() == 3:
        lab = lab[0]
    if lab.shape[0] != h or lab.shape[1] != w:
        lab_rs = F.interpolate(lab.unsqueeze(0).unsqueeze(0).float(), size=(h, w), mode='nearest')
        lab = lab_rs.squeeze(0).squeeze(0).long()

    feat_flat = feat.reshape(c, h * w).transpose(0, 1).numpy()  # (HW, C)
    lab_flat = lab.reshape(-1).numpy()  # (HW,)

    keep_mask = lab_flat != 255
    feat_flat = feat_flat[keep_mask]
    lab_flat = lab_flat[keep_mask]
    if feat_flat.shape[0] == 0:
        return None, None

    sampled_feat_list, sampled_lab_list = [], []
    for cls_id in np.unique(lab_flat):
        cls_idx = np.where(lab_flat == cls_id)[0]
        if cls_idx.size == 0:
            continue
        take_n = min(int(points_per_class), int(cls_idx.size))
        chosen = np.random.choice(cls_idx, size=take_n, replace=False)
        sampled_feat_list.append(feat_flat[chosen])
        sampled_lab_list.append(np.full((take_n,), cls_id, dtype=np.int64))

    if len(sampled_feat_list) == 0:
        return None, None

    return np.concatenate(sampled_feat_list, axis=0), np.concatenate(sampled_lab_list, axis=0)


def save_dataset_tsne(tsne_bank, save_dir, max_points_per_model=6000):
    if not _TSNE_AVAILABLE:
        return

    for model_tag, data in tsne_bank.items():
        if len(data['feats']) == 0:
            continue

        x = np.concatenate(data['feats'], axis=0)
        y = np.concatenate(data['labels'], axis=0)
        if x.shape[0] < 10:
            continue

        if x.shape[0] > int(max_points_per_model):
            idx = np.random.choice(x.shape[0], size=int(max_points_per_model), replace=False)
            x = x[idx]
            y = y[idx]

        perplexity = max(5, min(40, x.shape[0] // 20))
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=0)
        emb = tsne.fit_transform(x)

        plt.figure(figsize=(8, 7))
        uniq_cls = np.unique(y)
        cmap = plt.cm.get_cmap('tab20', 21)
        for cls_id in uniq_cls:
            m = y == cls_id
            color = cmap(int(cls_id) % 21)
            plt.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.65, color=color, label=str(int(cls_id)))

        plt.legend(title='VOC cls id', fontsize=8, ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'family': 'Times New Roman'})
        plt.title(f"{model_tag} dataset-level t-SNE (all classes)", fontfamily='Times New Roman')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dataset_tsne_{model_tag.lower()}.png"), dpi=220)
        plt.close()


def save_gradient_bar_plot(grad_stats, save_path):
    if len(grad_stats) == 0:
        return

    module_names = [name for name in grad_stats.keys() if not name.endswith("__count")]
    grad_means = [grad_stats[name] / max(1, grad_stats[f"{name}__count"]) for name in module_names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(module_names, grad_means, color="#4C72B0", edgecolor="black", linewidth=0.8)
    plt.ylabel("Mean backprop gradient", fontfamily='Times New Roman')
    plt.xlabel("Module", fontfamily='Times New Roman')
    plt.title("Gradient contribution analysis", fontfamily='Times New Roman')
    plt.xticks(rotation=20, ha='right', fontfamily='Times New Roman')
    plt.yticks(fontfamily='Times New Roman')
    for bar, value in zip(bars, grad_means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4e}",
                 ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _precision_recall_from_hist(hist):
    eps = 1e-6
    tp = np.diag(hist)
    precision = tp / (hist.sum(axis=0) + eps)
    recall = tp / (hist.sum(axis=1) + eps)
    valid = hist.sum(axis=1) > 0
    mean_precision = float(np.nanmean(precision[valid])) if np.any(valid) else 0.0
    mean_recall = float(np.nanmean(recall[valid])) if np.any(valid) else 0.0
    return mean_precision, mean_recall, precision, recall


def validate(model, dataset, test_scales=None):

    _preds, _gts, _msc_preds, cams = [], [], [], []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    model.cuda()
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    tsne_bank = {
        'SAM3': {'feats': [], 'labels': []},
        'DINO': {'feats': [], 'labels': []},
        'CLIP': {'feats': [], 'labels': []},
    }

    grad_bar_stats = {
        'SAM3_Fused': 0.0,
        'SAM3_Fused__count': 0,
        'DINO_Fused': 0.0,
        'DINO_Fused__count': 0,
        'CLIP_SegHead': 0.0,
        'CLIP_SegHead__count': 0,
        'DINO_SegHead': 0.0,
        'DINO_SegHead__count': 0,
    }

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num+=1

        name, inputs, labels, cls_labels = data
        names = name+name

        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs_for_vis = inputs.clone()

        _, _, h, w = inputs.shape
        ratio = cfg.clip_init.resize_long / max(h,w)
        _h, _w = int(h*ratio), int(w*ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        need_vis_payload = args.save_vis or args.save_grad_bar
        if need_vis_payload:
            segs_clip_cat, segs_dino_cat, cam, attn_loss, vis_payload = model(
                inputs_cat,
                names,
                mode='val',
                return_vis=args.save_vis,
                return_grad_stats=args.save_grad_bar,
            )
        else:
            segs_clip_cat, segs_dino_cat, cam, attn_loss = model(inputs_cat, names, mode='val')
            vis_payload = None

        segs_cat = 0.5 * segs_dino_cat + 0.5*segs_clip_cat
        
        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, s_h, s_w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_clip_cat, segs_dino_cat, cam_cat, attn_loss = model(inputs_cat, names, mode='val')

                segs_cat = 0.5* segs_dino_cat + 0.5*segs_clip_cat

                _segs_cat = F.interpolate(segs_cat, size=(s_h, s_w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)
        
        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)
        # print('seg_shape', seg_preds.shape, 'labels', labels.shape, 'cam', cam.shape)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))

        if args.save_vis and (args.vis_max_images < 0 or idx < args.vis_max_images):
            img_name = name[0]
            image_rgb = denormalize_to_uint8(inputs_for_vis[0])

            # Segmentation confidence heatmap
            seg_conf = torch.softmax(resized_msc_segs, dim=1)[0].max(dim=0)[0].detach().cpu().numpy()
            save_heatmap(seg_conf, os.path.join(args.work_dir, "vis", "heatmap", img_name + ".png"))

            # CAM score map and overlay on original image
            if vis_payload is not None and vis_payload.get('cam_score_maps', None) is not None:
                cam_score_map = vis_payload['cam_score_maps'][0].numpy()
            else:
                cam_score_map = cam[0].detach().cpu().numpy().astype(np.float32)
            if cam_score_map.shape[:2] != image_rgb.shape[:2]:
                cam_score_map = cv2.resize(cam_score_map, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            save_heatmap(cam_score_map, os.path.join(args.work_dir, "vis", "cam_heatmap", img_name + ".png"))
            save_overlay(image_rgb, cam_score_map, os.path.join(args.work_dir, "vis", "cam_overlay", img_name + ".png"))

            # CAM pseudo label and segmentation outputs
            cam_label = cam[0].detach().cpu().numpy().astype(np.uint8)
            imageio.imsave(os.path.join(args.work_dir, "vis", "cam_label", img_name + ".png"), cam_label)
            imageio.imsave(os.path.join(args.work_dir, "vis", "cam_label_cmap", img_name + ".png"), encode_cmap(cam_label).astype(np.uint8))

            seg_pred_np = seg_preds[0].detach().cpu().numpy().astype(np.uint8)
            msc_seg_pred_np = msc_seg_preds[0].detach().cpu().numpy().astype(np.uint8)
            imageio.imsave(os.path.join(args.work_dir, "vis", "seg", img_name + ".png"), seg_pred_np)
            imageio.imsave(os.path.join(args.work_dir, "vis", "seg_cmap", img_name + ".png"), encode_cmap(seg_pred_np).astype(np.uint8))
            imageio.imsave(os.path.join(args.work_dir, "vis", "msc_seg", img_name + ".png"), msc_seg_pred_np)
            imageio.imsave(os.path.join(args.work_dir, "vis", "msc_seg_cmap", img_name + ".png"), encode_cmap(msc_seg_pred_np).astype(np.uint8))

            # Different foundation model feature channel heatmaps
            if vis_payload is not None:
                sam3_fts = vis_payload.get('sam3_fts', None)
                dino_fts = vis_payload.get('dino_fts', None)
                clip_fts = vis_payload.get('clip_fts', None)

                save_feature_channel_grid(
                    sam3_fts[0] if sam3_fts is not None else None,
                    os.path.join(args.work_dir, "vis", "feature_channels", "sam3", img_name + ".png"),
                    topk=args.vis_topk_channels,
                )
                save_feature_channel_grid(
                    dino_fts[0] if dino_fts is not None else None,
                    os.path.join(args.work_dir, "vis", "feature_channels", "dino", img_name + ".png"),
                    topk=args.vis_topk_channels,
                )
                save_feature_channel_grid(
                    clip_fts[0] if clip_fts is not None else None,
                    os.path.join(args.work_dir, "vis", "feature_channels", "clip", img_name + ".png"),
                    topk=args.vis_topk_channels,
                )

                # Collect dataset-level t-SNE samples (all classes across all test images)
                for model_tag, model_feat in [('SAM3', sam3_fts), ('DINO', dino_fts), ('CLIP', clip_fts)]:
                    if model_feat is None:
                        continue
                    sampled_feats, sampled_labels = _sample_features_by_class(
                        model_feat[0],
                        labels[0],
                        points_per_class=args.tsne_points_per_class_per_image,
                    )
                    if sampled_feats is None:
                        continue
                    tsne_bank[model_tag]['feats'].append(sampled_feats)
                    tsne_bank[model_tag]['labels'].append(sampled_labels)

        if args.save_grad_bar and (args.grad_bar_max_images < 0 or idx < args.grad_bar_max_images):
            grad_module_tensors = [
                ('SAM3_Fused', vis_payload['sam3_fts'] if vis_payload is not None else None),
                ('DINO_Fused', vis_payload['dino_fts'] if vis_payload is not None else None),
                ('CLIP_SegHead', segs_clip_cat),
                ('DINO_SegHead', segs_dino_cat),
            ]

            grad_tensors = [tensor for _, tensor in grad_module_tensors if tensor is not None]
            grad_names = [name for name, tensor in grad_module_tensors if tensor is not None]
            if len(grad_tensors) > 0:
                fused_logits = 0.5 * (segs_clip_cat + segs_dino_cat)
                fused_logits = 0.5 * (fused_logits[0] + fused_logits[1].flip(-1))
                fused_logits = F.interpolate(fused_logits.unsqueeze(0), size=labels.shape[1:], mode='bilinear', align_corners=False)
                grad_loss = F.cross_entropy(fused_logits, labels.long(), ignore_index=cfg.dataset.ignore_index)
                grads = torch.autograd.grad(grad_loss, grad_tensors, retain_graph=False, allow_unused=True)

                for grad_name, grad_tensor in zip(grad_names, grads):
                    grad_value = float(grad_tensor.abs().mean().item()) if grad_tensor is not None else 0.0
                    grad_bar_stats[grad_name] += grad_value
                    grad_bar_stats[f"{grad_name}__count"] += 1


        if num % 1000 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            _preds, _gts, _msc_preds, cams = [], [], [], []


        np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})

    if args.save_vis:
        save_dataset_tsne(
            tsne_bank=tsne_bank,
            save_dir=os.path.join(args.work_dir, "vis", "tsne"),
            max_points_per_model=args.tsne_max_points_per_model,
        )

    if args.save_grad_bar:
        save_gradient_bar_plot(
            grad_stats=grad_bar_stats,
            save_path=os.path.join(args.work_dir, "vis", "grad_bar", "gradient_contribution_bar.png"),
        )
            
    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("crf post-processing...")

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 64
        bi_rgb_std=5,   # 5
        bi_w=4,         # 4
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.eval_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)
    hist = np.zeros((21, 21))
    hist, score = evaluate.scores(gts, preds, hist, 21)

    print(score)
    
    return True

def main(cfg):
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    model = RSL(num_classes=cfg.dataset.num_classes,
                     clip_model=cfg.clip_init.clip_pretrain_path,
                     dino_model=cfg.dino_init.dino_model,
                     dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
                     decoder_layers=cfg.dino_init.decoder_layer,
                     embedding_dim=cfg.clip_init.embedding_dim,
                     in_channels=cfg.clip_init.in_channels,
                     dataset_root_path=cfg.dataset.root_dir,
                     clip_flag=cfg.clip_init.clip_flag,
                     device='cuda')
    
    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(model=model, dataset=val_dataset, test_scales=[1, 1.5]) #[1, 0.75]
    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist)

    seg_mp, seg_mr, _, _ = _precision_recall_from_hist(preds_hist)
    msc_mp, msc_mr, _, _ = _precision_recall_from_hist(msc_preds_hist)
    cam_mp, cam_mr, _, _ = _precision_recall_from_hist(cams_hist)

    print("cams score:")
    print(cam_score)
    print(f"cams mean precision: {cam_mp:.4f}")
    print(f"cams mean recall   : {cam_mr:.4f}")
    print("cams confusion matrix:")
    print(cams_hist.astype(np.int64))
    print("segs score:")
    print(seg_score)
    print(f"segs mean precision: {seg_mp:.4f}")
    print(f"segs mean recall   : {seg_mr:.4f}")
    print("segs confusion matrix:")
    print(preds_hist.astype(np.int64))
    print("msc segs score:")
    print(msc_seg_score)
    print(f"msc segs mean precision: {msc_mp:.4f}")
    print(f"msc segs mean recall   : {msc_mr:.4f}")
    print("msc segs confusion matrix:")
    print(msc_preds_hist.astype(np.int64))

    # crf_proc(config=cfg)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    if args.save_vis:
        os.makedirs(args.work_dir + "/vis/heatmap", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/cam_heatmap", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/cam_overlay", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/cam_label", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/cam_label_cmap", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/seg", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/seg_cmap", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/msc_seg", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/msc_seg_cmap", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/feature_channels/sam3", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/feature_channels/dino", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/feature_channels/clip", exist_ok=True)
        os.makedirs(args.work_dir + "/vis/tsne", exist_ok=True)
    if args.save_grad_bar:
        os.makedirs(args.work_dir + "/vis/grad_bar", exist_ok=True)

    main(cfg=cfg)

