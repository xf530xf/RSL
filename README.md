

# RSL: Reliability-Aware and Self-Evolving Learning for Weakly Supervised Image Segmentation

Implementation of **RSL** for image-level weakly supervised semantic segmentation. It combines frozen CLIP, SAM3, and DINO features with reliability-aware supervision, an EMA teacher for progressive pseudo-label replacement, and boundary-aware distillation. The code includes PASCAL VOC 2012 and MS COCO 2014 training and evaluation, plus CUB-200-2011 and AquaOV255 experiment scripts.

Dataset files and pretrained/checkpoint weights are not included.

## Setup

Use a CUDA-capable PyTorch environment. Install PyTorch and torchvision for your CUDA version, then install the remaining code dependencies, including `transformers` with `Sam3Model`, `timm`, `omegaconf`, `tensorboard`, `pydensecrf`, `opencv-python`, `scikit-learn`, `joblib`, and `imageio`. The current `requirements.txt` contains legacy pins and may need adjustment for a modern SAM3/DINOv3 environment. Run all commands from the repository root.

Download CLIP ViT-B/16 weights from the [official CLIP release](https://github.com/openai/CLIP#model-card) and place them at `pretrained/ViT-B-16.pt`, or change `clip_init.clip_pretrain_path` in your config. The configured SAM3 model is `facebook/sam3`; DINOv3 is loaded through `timm`. Upstream model downloads may require network access and access approval.

## Data and configuration

| Dataset | Config | Training | Evaluation |
| --- | --- | --- | --- |
| PASCAL VOC 2012 | `configs/voc_attn_reg.yaml` | `scripts/dist_clip_voc.py` | `test_msc_flip_voc.py` |
| MS COCO 2014 | `configs/coco_attn_reg.yaml` | `scripts/dist_clip_coco.py` | `test_msc_flip_coco.py` |
| VOC with DINOv2 | `configs/voc_attn_reg_dinov2.yaml` | `scripts/dist_clip_voc_dinov2.py` | `test-VOC.py` |
| CUB-200-2011 | `configs/cub_attn_reg.yaml` | `scripts/train-cub.py` | `test-cub.py` |
| AquaOV255 | `configs/underwater_attn_reg.yaml` | `scripts/train-underwater.py` | `test-underwater.py` |

Set `dataset.root_dir` in the selected config to your dataset directory. The `dataset.name_list_dir` values point to the split/label metadata under `datasets/`; edit them if running from elsewhere. VOC uses the `train_aug` split and expects augmented segmentation labels. COCO expects images and VOC-style segmentation labels under the paths used by `datasets/coco.py`. For CUB and AquaOV255, see `scripts/prepare_cub_splits.py` and `scripts/prepare_underwater_splits.py`.

A typical VOC layout is:

```text
VOCdevkit/VOC2012/
├── JPEGImages/
├── ImageSets/Segmentation/
├── SegmentationClass/
└── SegmentationClassAug/
```

The network lives in the local `RSL/` Python package. CLIP, DINO, and SAM3 are upstream foundation models and retain their original names.

## Training

```bash
python scripts/dist_clip_voc.py --config configs/voc_attn_reg.yaml
python scripts/dist_clip_coco.py --config configs/coco_attn_reg.yaml
python scripts/dist_clip_voc_dinov2.py --config configs/voc_attn_reg_dinov2.yaml
```

Checkpoints are saved as `rsl_iter_<iteration>.pth` under the configured work directory's `checkpoints/<timestamp>/` folder. Training scripts currently set `CUDA_VISIBLE_DEVICES` in source; adjust the device there before running.

## Evaluation

Use a checkpoint generated with the matching config:

```bash
python test_msc_flip_voc.py --config configs/voc_attn_reg.yaml --model_path /path/to/rsl_iter_30000.pth
python test_msc_flip_coco.py --config configs/coco_attn_reg.yaml --model_path /path/to/rsl_iter_80000.pth
```

The iteration depends on your training run. The DINOv2, CUB, and AquaOV255 variants use their own evaluation scripts in the table above.

## Citation

Please cite the paper when its bibliographic record is available:

> Feng Xiao et al., “RSL: Reliability-Aware and Self-Evolving Learning for Weakly Supervised Image Segmentation.”

## Acknowledgements

This implementation builds on ideas and code from [Frozen CLIP-DINO / WeCLIP](https://github.com/zbf1991/WeCLIP), [AFA](https://github.com/rulixiang/afa), and [CLIP-ES](https://github.com/linyq2117/CLIP-ES). CLIP, SAM3, and DINO are third-party foundation models; refer to their projects for model access and licenses.

