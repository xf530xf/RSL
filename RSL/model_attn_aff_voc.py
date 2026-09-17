
# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .segformer_head import SegFormerHead
# import numpy as np
# import clip
# from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
# from pytorch_grad_cam import GradCAM
# from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
# import os
# from torchvision.transforms import Compose, Normalize
# from .Decoder.TransDecoder import DecoderTransformer
# from RSL.PAR import PAR
# from transformers import Sam3Model


# def Normalize_clip():
#     return Compose([
#     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


# def reshape_transform(tensor, height=28, width=28):
#     tensor = tensor.permute(1, 0, 2)
#     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result



# def zeroshot_classifier(classnames, templates, model):
#     with torch.no_grad():
#         zeroshot_weights = []
#         for classname in classnames:
#             texts = [template.format(classname) for template in templates] #format with class
#             texts = clip.tokenize(texts).cuda() #tokenize
#             class_embeddings = model.encode_text(texts) #embed with text encoder
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings.mean(dim=0)
#             class_embedding /= class_embedding.norm()
#             zeroshot_weights.append(class_embedding)
#         zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
#     return zeroshot_weights.t()


# def _refine_cams(ref_mod, images, cams, valid_key):
#     images = images.unsqueeze(0)
#     cams = cams.unsqueeze(0)
#     refined_cams = ref_mod(images.float(), cams.float())
#     refined_label = refined_cams.argmax(dim=1)
#     refined_label = valid_key[refined_label]

#     return refined_label.squeeze(0)


# class RSL(nn.Module):
#     def __init__(self, num_classes=None, clip_model=None, sam3_model='facebook/sam3',
#                  dino_model=None, dino_fts_dim=768, decoder_layers=3,
#                  embedding_dim=256, in_channels=None, dataset_root_path=None,
#                  clip_flag=16, dino_patch_size=16, device='cuda'):
#         super().__init__()
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
#         self.dino_fts_fuse_dim = dino_fts_dim
#         self.clip_flag = clip_flag
#         self.dino_patch_size = dino_patch_size

#         # CLIP encoder — kept frozen for text features + GradCAM only
#         self.encoder, _ = clip.load(clip_model, device=device)
#         for name, param in self.encoder.named_parameters():
#             if clip_flag == 14 and '23' not in name:
#                 param.requires_grad = False
#             if clip_flag == 16 and "11" not in name:
#                 param.requires_grad = False

#         # SAM3 ViT backbone — visual feature extractor (replaces CLIP visual)
#         _sam3_full = Sam3Model.from_pretrained(sam3_model)
#         self.sam3_encoder = _sam3_full.vision_encoder.backbone
#         del _sam3_full
#         for param in self.sam3_encoder.parameters():
#             param.requires_grad = False
#         self.sam3_patch_size = self.sam3_encoder.config.patch_size
#         self.sam3_window_size = self.sam3_encoder.config.window_size
#         self.sam3_hidden_size = self.sam3_encoder.config.hidden_size

#         # DINO encoder
#         self.use_timm_dino = True
#         self.dino_encoder = timm.create_model(
#                 dino_model,
#                 pretrained=True,
#                 num_classes=0,
#             )
#         self.dino_num_prefix_tokens = getattr(self.dino_encoder, 'num_prefix_tokens', 1)

#         for param in self.dino_encoder.parameters():
#             param.requires_grad = False

#         # in_channels should be [sam3_hidden_size]*4 = [1024,1024,1024,1024]
#         if in_channels is None:
#             in_channels = [self.sam3_hidden_size] * 4
#         self.in_channels = in_channels

#         self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
#                                               num_classes=self.num_classes, index=1)
        
#         self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
#                                               num_classes=self.num_classes, index=1)
        
#         self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)


#         self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
#                                                self.encoder)  # ['a rendering of a weird {}.'], model)
#         self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
#                                                self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


#         self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
#         self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
#         self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
#         self.cam_bg_thres = 1
#         self.encoder.eval()
#         self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
#         self.iter_num = 0
#         self.require_all_fts = True

#     def _forward_seg_branches(self, img):
#         b, _, h, w = img.shape

#         # SAM3 ViT features — single last hidden state
#         with torch.no_grad():
#             sam3_size = self.sam3_encoder.config.image_size
#             sam3_img = F.interpolate(img, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
#             sam3_out = self.sam3_encoder(pixel_values=sam3_img)
#             sam3_fts = sam3_out.last_hidden_state

#         sam3_spatial = sam3_size // self.sam3_patch_size
#         sam3_fts = sam3_fts.reshape(b, sam3_spatial, sam3_spatial, -1).permute(0, 3, 1, 2)
#         sam3_fts = F.interpolate(sam3_fts, size=(h // self.clip_flag, w // self.clip_flag),
#                                  mode='bilinear', align_corners=False)
#         fts = self.decoder_fts_fuse(sam3_fts.unsqueeze(0))
#         _, _, fts_h, fts_w = fts.shape

#         # DINO features
#         ps = self.dino_patch_size
#         with torch.no_grad():
#             dino_img_h, dino_img_w = (h // ps) * ps, (w // ps) * ps
#             dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
#             if self.use_timm_dino:
#                 dino_all = self.dino_encoder.forward_features(dino_img)
#                 skip = self.dino_num_prefix_tokens
#                 dino_fts = dino_all[:, skip:, :]
#             else:
#                 dino_out = self.dino_encoder(dino_img)
#                 skip = 1 + self.dino_num_register_tokens
#                 dino_fts = dino_out.last_hidden_state[:, skip:, :]
#             dino_h, dino_w = dino_img_h // ps, dino_img_w // ps

#         if isinstance(dino_fts, list):
#             for d_i, dino_fts_single in enumerate(dino_fts):
#                 dino_fts_single = dino_fts_single.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
#                 dino_fts[d_i] = dino_fts_single
#             dino_fts = torch.stack(dino_fts)
#             dino_fts = self.dino_decoder_fts_fuse(dino_fts)
#         else:
#             dino_fts = dino_fts.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
#             dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))

#         dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
#         seg_sam3, _ = self.decoder(fts)
#         seg_dino, _ = self.decoder(dino_fts)
#         return seg_sam3, seg_dino, fts, dino_fts

#     def forward_seg(self, img):
#         seg_sam3, seg_dino, _, _ = self._forward_seg_branches(img)
#         return seg_sam3, seg_dino


#     def get_param_groups(self):

#         param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

#         for param in list(self.decoder.parameters()):
#             param_groups[3].append(param)
#         for param in list(self.decoder_fts_fuse.parameters()):
#             param_groups[3].append(param)
#         for param in list(self.dino_decoder_fts_fuse.parameters()):
#             param_groups[3].append(param)

#         return param_groups
    


#     def forward(self, img, img_names='2007_000032', mode='train'):
#         cam_list = []
#         b, c, h, w = img.shape
#         self.encoder.eval()
#         self.iter_num += 1

#         # CLIP features — used only for GradCAM / CAM refinement
#         fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

#         fts_all_stack = torch.stack(fts_all, dim=0)
#         attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

#         if self.require_all_fts:
#             cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3)
#         else:
#             cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

#         seg_sam3, seg_dino, fts, dino_fts = self._forward_seg_branches(img)
#         _, _, fts_h, fts_w = fts.shape

#         sam3_dino_fts = torch.cat([fts, dino_fts], dim=1)

#         seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_sam3, dim=1)
#         seg_dino_prob = seg_dino_prob.detach()

#         attn_fts = F.interpolate(sam3_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
#         f_b, f_c, f_h, f_w = attn_fts.shape
#         attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
#         attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
#         attn_pred = torch.sigmoid(attn_pred)

#         for i, img_name in enumerate(img_names):
#             img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
#             img_i = img[i]
#             cam_fts = cam_fts_all[i]
#             cam_attn = attn_weight_stack[i]

#             seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

#             require_seg_trans = True
#             seg_dino_cam = seg_dino_prob[i]

#             # iter_w = 0.5

#             cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
#                                                                    self.bg_text_features, self.fg_text_features,
#                                                                    self.grad_cam,
#                                                                    mode=mode,
#                                                                    require_seg_trans=require_seg_trans,
#                                                                    seg_dino_cam=seg_dino_cam,
#                                                                    clip_flag = self.clip_flag
#                                                                           )

#             cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
#             cams = cam_dict['refined_cam'].cuda()

#             bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

#             cams = torch.cat([bg_score, cams], dim=0).cuda()
            
#             valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
#             valid_key = torch.from_numpy(valid_key).cuda()
            
#             with torch.no_grad():
#                 cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
#             cam_list.append(cam_labels)

#         all_cam_labels = torch.stack(cam_list, dim=0)

#         if self.training:
            # return seg_sam3, seg_dino, all_cam_labels, attn_pred
#         else:
#             return seg_sam3, seg_dino, all_cam_labels, attn_pred

        
    

# if __name__=="__main__":

#     pretrained_weights = torch.load('pretrained/mit_b1.pth')
#     rsl = RSL('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
#     rsl._param_groups()
#     dummy_input = torch.rand(2,3,512,512)
#     rsl(dummy_input)








# # import timm
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from .segformer_head import SegFormerHead
# # import numpy as np
# # import clip
# # from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
# # from pytorch_grad_cam import GradCAM
# # from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
# # import os
# # from torchvision.transforms import Compose, Normalize
# # from .Decoder.TransDecoder import DecoderTransformer
# # from RSL.PAR import PAR
# # from transformers import Sam3Model




# # def Normalize_clip():
# #     return Compose([
# #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


# # def reshape_transform(tensor, height=28, width=28):
# #     tensor = tensor.permute(1, 0, 2)
# #     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

# #     # Bring the channels to the first dimension,
# #     # like in CNNs.
# #     result = result.transpose(2, 3).transpose(1, 2)
# #     return result



# # def zeroshot_classifier(classnames, templates, model):
# #     with torch.no_grad():
# #         zeroshot_weights = []
# #         for classname in classnames:
# #             texts = [template.format(classname) for template in templates] #format with class
# #             texts = clip.tokenize(texts).cuda() #tokenize
# #             class_embeddings = model.encode_text(texts) #embed with text encoder
# #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
# #             class_embedding = class_embeddings.mean(dim=0)
# #             class_embedding /= class_embedding.norm()
# #             zeroshot_weights.append(class_embedding)
# #         zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
# #     return zeroshot_weights.t()


# # def _refine_cams(ref_mod, images, cams, valid_key):
# #     images = images.unsqueeze(0)
# #     cams = cams.unsqueeze(0)
# #     refined_cams = ref_mod(images.float(), cams.float())
# #     refined_label = refined_cams.argmax(dim=1)
# #     refined_label = valid_key[refined_label]

# #     return refined_label.squeeze(0)


# # class RSL(nn.Module):
# #     def __init__(self, num_classes=None, clip_model=None, sam3_model='facebook/sam3',
# #                  dino_model=None, dino_fts_dim=768, decoder_layers=3,
# #                  embedding_dim=256, in_channels=None, dataset_root_path=None,
# #                  clip_flag=16, dino_patch_size=16, device='cuda'):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         self.embedding_dim = embedding_dim
# #         self.dino_fts_fuse_dim = dino_fts_dim
# #         self.clip_flag = clip_flag
# #         self.dino_patch_size = dino_patch_size

# #         # CLIP encoder — kept frozen for text features + GradCAM only
# #         self.encoder, _ = clip.load(clip_model, device=device)
# #         for name, param in self.encoder.named_parameters():
# #             if clip_flag == 14 and '23' not in name:
# #                 param.requires_grad = False
# #             if clip_flag == 16 and "11" not in name:
# #                 param.requires_grad = False

# #         # SAM3 ViT backbone — visual feature extractor (replaces CLIP visual)
# #         _sam3_full = Sam3Model.from_pretrained(sam3_model)
# #         self.sam3_encoder = _sam3_full.vision_encoder.backbone
# #         del _sam3_full
# #         for param in self.sam3_encoder.parameters():
# #             param.requires_grad = False
# #         self.sam3_patch_size = self.sam3_encoder.config.patch_size
# #         self.sam3_window_size = self.sam3_encoder.config.window_size
# #         self.sam3_hidden_size = self.sam3_encoder.config.hidden_size

# #         # DINO encoder
# #         self.use_timm_dino = True
# #         self.dino_encoder = timm.create_model(
# #                 dino_model,
# #                 pretrained=True,
# #                 num_classes=0,
# #             )
# #         self.dino_num_prefix_tokens = getattr(self.dino_encoder, 'num_prefix_tokens', 1)

# #         for param in self.dino_encoder.parameters():
# #             param.requires_grad = False

# #         # in_channels should be [sam3_hidden_size]*4 = [1024,1024,1024,1024]
# #         if in_channels is None:
# #             in_channels = [self.sam3_hidden_size] * 4
# #         self.in_channels = in_channels

# #         self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
# #                                               num_classes=self.num_classes, index=1)
        
# #         self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
# #                                               num_classes=self.num_classes, index=1)
        
# #         self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)


# #         self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
# #                                                self.encoder)  # ['a rendering of a weird {}.'], model)
# #         self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
# #                                                self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


# #         self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
# #         self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
# #         self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
# #         self.cam_bg_thres = 1
# #         self.encoder.eval()
# #         self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
# #         self.iter_num = 0
# #         self.require_all_fts = True


# #     def get_param_groups(self):

# #         param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

# #         for param in list(self.decoder.parameters()):
# #             param_groups[3].append(param)
# #         for param in list(self.decoder_fts_fuse.parameters()):
# #             param_groups[3].append(param)
# #         for param in list(self.dino_decoder_fts_fuse.parameters()):
# #             param_groups[3].append(param)

# #         return param_groups
    


# #     def forward(self, img, img_names='2007_000032', mode='train'):
# #         cam_list = []
# #         b, c, h, w = img.shape
# #         self.encoder.eval()
# #         self.iter_num += 1

# #         # CLIP features — used only for GradCAM / CAM refinement
# #         fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

# #         fts_all_stack = torch.stack(fts_all, dim=0)
# #         attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

# #         if self.require_all_fts:
# #             cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3)
# #         else:
# #             cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

# #         # SAM3 ViT features — visual backbone for SegFormerHead decoder
# #         sam3_size = self.sam3_encoder.config.image_size
# #         with torch.no_grad():
# #             sam3_img = F.interpolate(img, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
# #             sam3_out = self.sam3_encoder(pixel_values=sam3_img)
# #             sam3_fts = sam3_out.last_hidden_state

# #         sam3_spatial = sam3_size // self.sam3_patch_size
# #         sam3_fts = sam3_fts.reshape(b, sam3_spatial, sam3_spatial, -1).permute(0, 3, 1, 2)
# #         sam3_fts = F.interpolate(sam3_fts, size=(h // self.clip_flag, w // self.clip_flag),
# #                                  mode='bilinear', align_corners=False)

# #         fts = self.decoder_fts_fuse(sam3_fts.unsqueeze(0))
# #         _, _, fts_h, fts_w = fts.shape

# #         # DINO features
# #         ps = self.dino_patch_size
# #         with torch.no_grad():
# #             dino_img_h, dino_img_w = (h // ps) * ps, (w // ps) * ps
# #             dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
# #             if self.use_timm_dino:
# #                 dino_all = self.dino_encoder.forward_features(dino_img)
# #                 skip = self.dino_num_prefix_tokens
# #                 dino_fts = dino_all[:, skip:, :]
# #             else:
# #                 dino_out = self.dino_encoder(dino_img)
# #                 skip = 1 + self.dino_num_register_tokens
# #                 dino_fts = dino_out.last_hidden_state[:, skip:, :]
# #             dino_h, dino_w = dino_img_h // ps, dino_img_w // ps

# #         if isinstance(dino_fts, list):
# #             for d_i, dino_fts_single in enumerate(dino_fts):
# #                 dino_fts_single = dino_fts_single.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
# #                 dino_fts[d_i] = dino_fts_single

# #             dino_fts = torch.stack(dino_fts)
# #             dino_fts = self.dino_decoder_fts_fuse(dino_fts)

# #         else:
# #             dino_fts = dino_fts.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
# #             dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
# #         dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

# #         seg_sam3, _ = self.decoder(fts)
# #         seg_dino, _ = self.decoder(dino_fts)

# #         sam3_dino_fts = torch.cat([fts, dino_fts], dim=1)

# #         seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_sam3, dim=1)
# #         seg_dino_prob = seg_dino_prob.detach()

# #         attn_fts = F.interpolate(sam3_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
# #         f_b, f_c, f_h, f_w = attn_fts.shape
# #         attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
# #         attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
# #         attn_pred = torch.sigmoid(attn_pred)

# #         for i, img_name in enumerate(img_names):
# #             img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
# #             img_i = img[i]
# #             cam_fts = cam_fts_all[i]
# #             cam_attn = attn_weight_stack[i]

# #             seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

# #             require_seg_trans = True
# #             seg_dino_cam = seg_dino_prob[i]

# #             # iter_w = 0.5

# #             cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
# #                                                                    self.bg_text_features, self.fg_text_features,
# #                                                                    self.grad_cam,
# #                                                                    mode=mode,
# #                                                                    require_seg_trans=require_seg_trans,
# #                                                                    seg_dino_cam=seg_dino_cam,
# #                                                                    clip_flag = self.clip_flag
# #                                                                           )

# #             cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
# #             cams = cam_dict['refined_cam'].cuda()

# #             bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

# #             cams = torch.cat([bg_score, cams], dim=0).cuda()
            
# #             valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
# #             valid_key = torch.from_numpy(valid_key).cuda()
            
# #             with torch.no_grad():
# #                 cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
# #             cam_list.append(cam_labels)

# #         all_cam_labels = torch.stack(cam_list, dim=0)

# #         if self.training:
# #             return seg_sam3, seg_dino, all_cam_labels, attn_pred
# #         else:
# #             return seg_sam3, seg_dino, all_cam_labels, attn_pred

        
    

# # if __name__=="__main__":

# #     pretrained_weights = torch.load('pretrained/mit_b1.pth')
# #     rsl = RSL('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
# #     rsl._param_groups()
# #     dummy_input = torch.rand(2,3,512,512)
# #     rsl(dummy_input)































# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .segformer_head import SegFormerHead
# import numpy as np
# import clip
# from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
# from pytorch_grad_cam import GradCAM
# from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
# import os
# from torchvision.transforms import Compose, Normalize
# from .Decoder.TransDecoder import DecoderTransformer
# from RSL.PAR import PAR
# from transformers import Sam3Model


# def Normalize_clip():
#     return Compose([
#     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


# def reshape_transform(tensor, height=28, width=28):
#     tensor = tensor.permute(1, 0, 2)
#     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result



# def zeroshot_classifier(classnames, templates, model):
#     with torch.no_grad():
#         zeroshot_weights = []
#         for classname in classnames:
#             texts = [template.format(classname) for template in templates] #format with class
#             texts = clip.tokenize(texts).cuda() #tokenize
#             class_embeddings = model.encode_text(texts) #embed with text encoder
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings.mean(dim=0)
#             class_embedding /= class_embedding.norm()
#             zeroshot_weights.append(class_embedding)
#         zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
#     return zeroshot_weights.t()


# def _refine_cams(ref_mod, images, cams, valid_key):
#     images = images.unsqueeze(0)
#     cams = cams.unsqueeze(0)
#     refined_cams = ref_mod(images.float(), cams.float())
#     refined_label = refined_cams.argmax(dim=1)
#     refined_label = valid_key[refined_label]

#     return refined_label.squeeze(0)


# class RSL(nn.Module):
#     def __init__(self, num_classes=None, clip_model=None, sam3_model='facebook/sam3',
#                  dino_model=None, dino_fts_dim=768, decoder_layers=3,
#                  embedding_dim=256, in_channels=None, dataset_root_path=None,
#                  clip_flag=16, dino_patch_size=16, device='cuda'):
#         super().__init__()
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
#         self.dino_fts_fuse_dim = dino_fts_dim
#         self.clip_flag = clip_flag
#         self.dino_patch_size = dino_patch_size

#         # CLIP encoder — kept frozen for text features + GradCAM only
#         self.encoder, _ = clip.load(clip_model, device=device)
#         for name, param in self.encoder.named_parameters():
#             if clip_flag == 14 and '23' not in name:
#                 param.requires_grad = False
#             if clip_flag == 16 and "11" not in name:
#                 param.requires_grad = False

#         # SAM3 ViT backbone — visual feature extractor (replaces CLIP visual)
#         _sam3_full = Sam3Model.from_pretrained(sam3_model)
#         self.sam3_encoder = _sam3_full.vision_encoder.backbone
#         del _sam3_full
#         for param in self.sam3_encoder.parameters():
#             param.requires_grad = False
#         self.sam3_patch_size = self.sam3_encoder.config.patch_size
#         self.sam3_window_size = self.sam3_encoder.config.window_size
#         self.sam3_hidden_size = self.sam3_encoder.config.hidden_size

#         # DINO encoder
#         self.use_timm_dino = True
#         self.dino_encoder = timm.create_model(
#                 dino_model,
#                 pretrained=True,
#                 num_classes=0,
#             )
#         self.dino_num_prefix_tokens = getattr(self.dino_encoder, 'num_prefix_tokens', 1)

#         for param in self.dino_encoder.parameters():
#             param.requires_grad = False

#         # in_channels should be [sam3_hidden_size]*4 = [1024,1024,1024,1024]
#         if in_channels is None:
#             in_channels = [self.sam3_hidden_size] * 4
#         self.in_channels = in_channels

#         self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
#                                               num_classes=self.num_classes, index=1)
        
#         self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
#                                               num_classes=self.num_classes, index=1)
        
#         self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)


#         self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
#                                                self.encoder)  # ['a rendering of a weird {}.'], model)
#         self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
#                                                self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


#         self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
#         self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
#         self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
#         self.cam_bg_thres = 1
#         self.encoder.eval()
#         self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
#         self.iter_num = 0
#         self.require_all_fts = True


#     def get_param_groups(self):

#         param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

#         for param in list(self.decoder.parameters()):
#             param_groups[3].append(param)
#         for param in list(self.decoder_fts_fuse.parameters()):
#             param_groups[3].append(param)
#         for param in list(self.dino_decoder_fts_fuse.parameters()):
#             param_groups[3].append(param)

#         return param_groups
    


#     def forward(self, img, img_names='2007_000032', mode='train'):
#         cam_list = []
#         b, c, h, w = img.shape
#         self.encoder.eval()
#         self.iter_num += 1

#         # CLIP features — used only for GradCAM / CAM refinement
#         fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

#         fts_all_stack = torch.stack(fts_all, dim=0)
#         attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

#         if self.require_all_fts:
#             cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3)
#         else:
#             cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

#         # SAM3 ViT features — visual backbone for SegFormerHead decoder
#         sam3_size = self.sam3_encoder.config.image_size
#         with torch.no_grad():
#             sam3_img = F.interpolate(img, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
#             sam3_out = self.sam3_encoder(pixel_values=sam3_img)
#             sam3_fts = sam3_out.last_hidden_state

#         sam3_spatial = sam3_size // self.sam3_patch_size
#         sam3_fts = sam3_fts.reshape(b, sam3_spatial, sam3_spatial, -1).permute(0, 3, 1, 2)
#         sam3_fts = F.interpolate(sam3_fts, size=(h // self.clip_flag, w // self.clip_flag),
#                                  mode='bilinear', align_corners=False)

#         fts = self.decoder_fts_fuse(sam3_fts.unsqueeze(0))
#         _, _, fts_h, fts_w = fts.shape

#         # DINO features
#         ps = self.dino_patch_size
#         with torch.no_grad():
#             dino_img_h, dino_img_w = (h // ps) * ps, (w // ps) * ps
#             dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
#             if self.use_timm_dino:
#                 dino_all = self.dino_encoder.forward_features(dino_img)
#                 skip = self.dino_num_prefix_tokens
#                 dino_fts = dino_all[:, skip:, :]
#             else:
#                 dino_out = self.dino_encoder(dino_img)
#                 skip = 1 + self.dino_num_register_tokens
#                 dino_fts = dino_out.last_hidden_state[:, skip:, :]
#             dino_h, dino_w = dino_img_h // ps, dino_img_w // ps

#         if isinstance(dino_fts, list):
#             for d_i, dino_fts_single in enumerate(dino_fts):
#                 dino_fts_single = dino_fts_single.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
#                 dino_fts[d_i] = dino_fts_single

#             dino_fts = torch.stack(dino_fts)
#             dino_fts = self.dino_decoder_fts_fuse(dino_fts)

#         else:
#             dino_fts = dino_fts.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
#             dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
#         dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

#         seg_sam3, _ = self.decoder(fts)
#         seg_dino, _ = self.decoder(dino_fts)

#         sam3_dino_fts = torch.cat([fts, dino_fts], dim=1)

#         seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_sam3, dim=1)
#         seg_dino_prob = seg_dino_prob.detach()

#         attn_fts = F.interpolate(sam3_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
#         f_b, f_c, f_h, f_w = attn_fts.shape
#         attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
#         attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
#         attn_pred = torch.sigmoid(attn_pred)

#         for i, img_name in enumerate(img_names):
#             img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
#             img_i = img[i]
#             cam_fts = cam_fts_all[i]
#             cam_attn = attn_weight_stack[i]

#             seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

#             require_seg_trans = True
#             seg_dino_cam = seg_dino_prob[i]

#             # iter_w = 0.5

#             cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
#                                                                    self.bg_text_features, self.fg_text_features,
#                                                                    self.grad_cam,
#                                                                    mode=mode,
#                                                                    require_seg_trans=require_seg_trans,
#                                                                    seg_dino_cam=seg_dino_cam,
#                                                                    clip_flag = self.clip_flag
#                                                                           )

#             cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
#             cams = cam_dict['refined_cam'].cuda()

#             bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

#             cams = torch.cat([bg_score, cams], dim=0).cuda()
            
#             valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
#             valid_key = torch.from_numpy(valid_key).cuda()
            
#             with torch.no_grad():
#                 cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
#             cam_list.append(cam_labels)

#         all_cam_labels = torch.stack(cam_list, dim=0)

#         if self.training:
#             return seg_sam3, seg_dino, all_cam_labels, attn_pred
#         else:
#             return seg_sam3, seg_dino, all_cam_labels, attn_pred

        
    

# if __name__=="__main__":

#     pretrained_weights = torch.load('pretrained/mit_b1.pth')
#     rsl = RSL('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
#     rsl._param_groups()
#     dummy_input = torch.rand(2,3,512,512)
#     rsl(dummy_input)


















































































# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .segformer_head import SegFormerHead
# import numpy as np
# import clip
# from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
# from pytorch_grad_cam import GradCAM
# from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
# import os
# from torchvision.transforms import Compose, Normalize
# from .Decoder.TransDecoder import DecoderTransformer
# from RSL.PAR import PAR
# from transformers import Sam3Model


# def Normalize_clip():
#     return Compose([
#     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


# def reshape_transform(tensor, height=28, width=28):
#     tensor = tensor.permute(1, 0, 2)
#     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result



# def zeroshot_classifier(classnames, templates, model):
#     with torch.no_grad():
#         zeroshot_weights = []
#         for classname in classnames:
#             texts = [template.format(classname) for template in templates] #format with class
#             texts = clip.tokenize(texts).cuda() #tokenize
#             class_embeddings = model.encode_text(texts) #embed with text encoder
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings.mean(dim=0)
#             class_embedding /= class_embedding.norm()
#             zeroshot_weights.append(class_embedding)
#         zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
#     return zeroshot_weights.t()


# def _refine_cams(ref_mod, images, cams, valid_key):
#     images = images.unsqueeze(0)
#     cams = cams.unsqueeze(0)
#     refined_cams = ref_mod(images.float(), cams.float())
#     refined_label = refined_cams.argmax(dim=1)
#     refined_label = valid_key[refined_label]

#     return refined_label.squeeze(0)


# class RSL(nn.Module):
#     def __init__(self, num_classes=None, clip_model=None, sam3_model='facebook/sam3',
#                  dino_model=None, dino_fts_dim=768, decoder_layers=3,
#                  embedding_dim=256, in_channels=None, dataset_root_path=None,
#                  clip_flag=16, dino_patch_size=16, device='cuda'):
#         super().__init__()
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
#         self.dino_fts_fuse_dim = dino_fts_dim
#         self.clip_flag = clip_flag
#         self.dino_patch_size = dino_patch_size

#         # CLIP encoder — kept frozen for text features + GradCAM only
#         self.encoder, _ = clip.load(clip_model, device=device)
#         for name, param in self.encoder.named_parameters():
#             if clip_flag == 14 and '23' not in name:
#                 param.requires_grad = False
#             if clip_flag == 16 and "11" not in name:
#                 param.requires_grad = False

#         # SAM3 ViT backbone — visual feature extractor (replaces CLIP visual)
#         _sam3_full = Sam3Model.from_pretrained(sam3_model)
#         self.sam3_encoder = _sam3_full.vision_encoder.backbone
#         del _sam3_full
#         for param in self.sam3_encoder.parameters():
#             param.requires_grad = False
#         self.sam3_patch_size = self.sam3_encoder.config.patch_size
#         self.sam3_window_size = self.sam3_encoder.config.window_size
#         self.sam3_hidden_size = self.sam3_encoder.config.hidden_size

#         # DINO encoder
#         self.use_timm_dino = True
#         self.dino_encoder = timm.create_model(
#                 dino_model,
#                 pretrained=True,
#                 num_classes=0,
#             )
#         self.dino_num_prefix_tokens = getattr(self.dino_encoder, 'num_prefix_tokens', 1)

#         for param in self.dino_encoder.parameters():
#             param.requires_grad = False

#         # in_channels should be [sam3_hidden_size]*4 = [1024,1024,1024,1024]
#         if in_channels is None:
#             in_channels = [self.sam3_hidden_size] * 4
#         self.in_channels = in_channels

#         self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
#                                               num_classes=self.num_classes, index=1)
        
#         self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
#                                               num_classes=self.num_classes, index=1)
        
#         self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)


#         self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
#                                                self.encoder)  # ['a rendering of a weird {}.'], model)
#         self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
#                                                self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


#         self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
#         self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
#         self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
#         self.cam_bg_thres = 1
#         self.encoder.eval()
#         self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
#         self.iter_num = 0
#         self.require_all_fts = True

#     def _forward_seg_branches(self, img):
#         b, _, h, w = img.shape

#         # SAM3 ViT features — single last hidden state
#         with torch.no_grad():
#             sam3_size = self.sam3_encoder.config.image_size
#             sam3_img = F.interpolate(img, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
#             sam3_out = self.sam3_encoder(pixel_values=sam3_img)
#             sam3_fts = sam3_out.last_hidden_state

#         sam3_spatial = sam3_size // self.sam3_patch_size
#         sam3_fts = sam3_fts.reshape(b, sam3_spatial, sam3_spatial, -1).permute(0, 3, 1, 2)
#         sam3_fts = F.interpolate(sam3_fts, size=(h // self.clip_flag, w // self.clip_flag),
#                                  mode='bilinear', align_corners=False)
#         fts = self.decoder_fts_fuse(sam3_fts.unsqueeze(0))
#         _, _, fts_h, fts_w = fts.shape

#         # DINO features
#         ps = self.dino_patch_size
#         with torch.no_grad():
#             dino_img_h, dino_img_w = (h // ps) * ps, (w // ps) * ps
#             dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
#             if self.use_timm_dino:
#                 dino_all = self.dino_encoder.forward_features(dino_img)
#                 skip = self.dino_num_prefix_tokens
#                 dino_fts = dino_all[:, skip:, :]
#             else:
#                 dino_out = self.dino_encoder(dino_img)
#                 skip = 1 + self.dino_num_register_tokens
#                 dino_fts = dino_out.last_hidden_state[:, skip:, :]
#             dino_h, dino_w = dino_img_h // ps, dino_img_w // ps

#         if isinstance(dino_fts, list):
#             for d_i, dino_fts_single in enumerate(dino_fts):
#                 dino_fts_single = dino_fts_single.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
#                 dino_fts[d_i] = dino_fts_single
#             dino_fts = torch.stack(dino_fts)
#             dino_fts = self.dino_decoder_fts_fuse(dino_fts)
#         else:
#             dino_fts = dino_fts.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
#             dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))

#         dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
#         seg_sam3, _ = self.decoder(fts)
#         seg_dino, _ = self.decoder(dino_fts)
#         return seg_sam3, seg_dino, fts, dino_fts

#     def forward_seg(self, img):
#         seg_sam3, seg_dino, _, _ = self._forward_seg_branches(img)
#         return seg_sam3, seg_dino


#     def get_param_groups(self):

#         param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

#         for param in list(self.decoder.parameters()):
#             param_groups[3].append(param)
#         for param in list(self.decoder_fts_fuse.parameters()):
#             param_groups[3].append(param)
#         for param in list(self.dino_decoder_fts_fuse.parameters()):
#             param_groups[3].append(param)

#         return param_groups
    


#     def forward(self, img, img_names='2007_000032', mode='train'):
#         cam_list = []
#         b, c, h, w = img.shape
#         self.encoder.eval()
#         self.iter_num += 1

#         # CLIP features — used only for GradCAM / CAM refinement
#         fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

#         fts_all_stack = torch.stack(fts_all, dim=0)
#         attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

#         if self.require_all_fts:
#             cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3)
#         else:
#             cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

#         seg_sam3, seg_dino, fts, dino_fts = self._forward_seg_branches(img)
#         _, _, fts_h, fts_w = fts.shape

#         sam3_dino_fts = torch.cat([fts, dino_fts], dim=1)

#         seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_sam3, dim=1)
#         seg_dino_prob = seg_dino_prob.detach()

#         attn_fts = F.interpolate(sam3_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
#         f_b, f_c, f_h, f_w = attn_fts.shape
#         attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
#         attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
#         attn_pred = torch.sigmoid(attn_pred)

#         for i, img_name in enumerate(img_names):
#             img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
#             img_i = img[i]
#             cam_fts = cam_fts_all[i]
#             cam_attn = attn_weight_stack[i]

#             seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

#             require_seg_trans = True
#             seg_dino_cam = seg_dino_prob[i]

#             # iter_w = 0.5

#             cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
#                                                                    self.bg_text_features, self.fg_text_features,
#                                                                    self.grad_cam,
#                                                                    mode=mode,
#                                                                    require_seg_trans=require_seg_trans,
#                                                                    seg_dino_cam=seg_dino_cam,
#                                                                    clip_flag = self.clip_flag
#                                                                           )

#             cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
#             cams = cam_dict['refined_cam'].cuda()

#             bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

#             cams = torch.cat([bg_score, cams], dim=0).cuda()
            
#             valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
#             valid_key = torch.from_numpy(valid_key).cuda()
            
#             with torch.no_grad():
#                 cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
#             cam_list.append(cam_labels)

#         all_cam_labels = torch.stack(cam_list, dim=0)

#         if self.training:
#             return seg_sam3, seg_dino, all_cam_labels, attn_pred, fts, dino_fts
#         else:
#             return seg_sam3, seg_dino, all_cam_labels, attn_pred

        
    

# if __name__=="__main__":

#     pretrained_weights = torch.load('pretrained/mit_b1.pth')
#     rsl = RSL('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
#     rsl._param_groups()
#     dummy_input = torch.rand(2,3,512,512)
#     rsl(dummy_input)












import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from RSL.PAR import PAR
from transformers import Sam3Model


def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)
    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class TextConditionedModulation(nn.Module):
    """
    Text-Conditioned Feature Modulation (TCFM).

    Injects CLIP class-level text semantics into visual features through:
      1) per-class spatial relevance maps (text-visual cosine similarity)
      2) image-adaptive FiLM (Feature-wise Linear Modulation) from text context
      3) residual gating for training stability
    """
    def __init__(self, visual_dim=256, text_dim=512, num_classes=21):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)

        self.relevance_conv = nn.Sequential(
            nn.Conv2d(num_classes, visual_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(visual_dim // 4, visual_dim, 1),
        )

        self.gamma_gen = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid()
        )
        self.beta_gen = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.Tanh()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(visual_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_fts, text_fts):
        """
        Args:
            visual_fts: (B, C, H, W) features from SegFormerHead
            text_fts:   (num_classes, text_dim) frozen CLIP text embeddings
        Returns:
            modulated features with the same shape (B, C, H, W)
        """
        b, c, h, w = visual_fts.shape

        # Keep Linear input dtype aligned with layer weights, then match visual dtype.
        text_fts = text_fts.to(device=visual_fts.device, dtype=self.text_proj.weight.dtype)
        text_emb = self.text_proj(text_fts)                          # (K, C)
        if text_emb.dtype != visual_fts.dtype:
            text_emb = text_emb.to(dtype=visual_fts.dtype)

        vis_norm = F.normalize(visual_fts, dim=1)
        txt_norm = F.normalize(text_emb, dim=1)
        spatial_sim = torch.einsum('bchw,kc->bkhw', vis_norm, txt_norm)  # (B, K, H, W)

        text_spatial_cue = self.relevance_conv(spatial_sim)              # (B, C, H, W)

        text_context = F.softmax(spatial_sim.mean(dim=[2, 3]), dim=1)    # (B, K)
        text_pool = torch.mm(text_context, text_emb)                     # (B, C)

        gamma = self.gamma_gen(text_pool).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        beta  = self.beta_gen(text_pool).unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)

        modulated = gamma * visual_fts + beta + text_spatial_cue

        gate_weight = self.gate(torch.cat([visual_fts, modulated], dim=1))  # (B, 1, H, W)
        return gate_weight * modulated + (1 - gate_weight) * visual_fts


class RSL(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, sam3_model='facebook/sam3',
                 dino_model=None, dino_fts_dim=768, decoder_layers=3,
                 embedding_dim=256, in_channels=None, dataset_root_path=None,
                 clip_flag=16, dino_patch_size=16, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = dino_fts_dim
        self.clip_flag = clip_flag
        self.dino_patch_size = dino_patch_size

        # CLIP encoder — kept frozen for text features + GradCAM only
        self.encoder, _ = clip.load(clip_model, device=device)
        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad = False
            if clip_flag == 16 and "11" not in name:
                param.requires_grad = False

        # SAM3 ViT backbone — visual feature extractor (replaces CLIP visual)
        _sam3_full = Sam3Model.from_pretrained(sam3_model)
        self.sam3_encoder = _sam3_full.vision_encoder.backbone
        del _sam3_full
        for param in self.sam3_encoder.parameters():
            param.requires_grad = False
        self.sam3_patch_size = self.sam3_encoder.config.patch_size
        self.sam3_window_size = self.sam3_encoder.config.window_size
        self.sam3_hidden_size = self.sam3_encoder.config.hidden_size

        # DINO encoder
        self.use_timm_dino = True
        self.dino_encoder = timm.create_model(
                dino_model,
                pretrained=True,
                num_classes=0,
            )
        self.dino_num_prefix_tokens = getattr(self.dino_encoder, 'num_prefix_tokens', 1)

        for param in self.dino_encoder.parameters():
            param.requires_grad = False

        # in_channels should be [sam3_hidden_size]*4 = [1024,1024,1024,1024]
        if in_channels is None:
            in_channels = [self.sam3_hidden_size] * 4
        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)


        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)

        # TCFM: merge bg (averaged) + fg text embeddings → (num_classes, text_dim)
        text_dim = self.fg_text_features.shape[1]
        bg_mean = self.bg_text_features.mean(dim=0, keepdim=True)
        self.register_buffer('all_text_features',
                             torch.cat([bg_mean, self.fg_text_features], dim=0))
        self.text_modulation = TextConditionedModulation(
            visual_dim=self.embedding_dim,
            text_dim=text_dim,
            num_classes=self.num_classes,
        )

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
        self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
        self.iter_num = 0
        self.require_all_fts = True

    def _forward_seg_branches(self, img):
        b, _, h, w = img.shape

        # SAM3 ViT features — single last hidden state
        with torch.no_grad():
            sam3_size = self.sam3_encoder.config.image_size
            sam3_img = F.interpolate(img, size=(sam3_size, sam3_size), mode='bilinear', align_corners=False)
            sam3_out = self.sam3_encoder(pixel_values=sam3_img)
            sam3_fts = sam3_out.last_hidden_state

        sam3_spatial = sam3_size // self.sam3_patch_size
        sam3_fts = sam3_fts.reshape(b, sam3_spatial, sam3_spatial, -1).permute(0, 3, 1, 2)
        sam3_fts = F.interpolate(sam3_fts, size=(h // self.clip_flag, w // self.clip_flag),
                                 mode='bilinear', align_corners=False)
        fts = self.decoder_fts_fuse(sam3_fts.unsqueeze(0))
        _, _, fts_h, fts_w = fts.shape

        # DINO features
        ps = self.dino_patch_size
        with torch.no_grad():
            dino_img_h, dino_img_w = (h // ps) * ps, (w // ps) * ps
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            if self.use_timm_dino:
                dino_all = self.dino_encoder.forward_features(dino_img)
                skip = self.dino_num_prefix_tokens
                dino_fts = dino_all[:, skip:, :]
            else:
                dino_out = self.dino_encoder(dino_img)
                skip = 1 + self.dino_num_register_tokens
                dino_fts = dino_out.last_hidden_state[:, skip:, :]
            dino_h, dino_w = dino_img_h // ps, dino_img_w // ps

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts_single = dino_fts_single.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
                dino_fts[d_i] = dino_fts_single
            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
        else:
            dino_fts = dino_fts.reshape([b, dino_h, dino_w, -1]).permute(0, 3, 1, 2)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))

        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

        # TCFM: inject class-level text semantics into both visual branches
        fts = self.text_modulation(fts, self.all_text_features)
        dino_fts = self.text_modulation(dino_fts, self.all_text_features)

        seg_sam3, _ = self.decoder(fts)
        seg_dino, _ = self.decoder(dino_fts)
        return seg_sam3, seg_dino, fts, dino_fts

    def forward_seg(self, img):
        seg_sam3, seg_dino, _, _ = self._forward_seg_branches(img)
        return seg_sam3, seg_dino


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.text_modulation.parameters()):
            param_groups[3].append(param)

        return param_groups
    


    def forward(self, img, img_names='2007_000032', mode='train', return_vis=False, return_grad_stats=False):
        cam_list = []
        cam_score_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        # CLIP features — used only for GradCAM / CAM refinement
        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        fts_all_stack = torch.stack(fts_all, dim=0)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

        if self.require_all_fts:
            cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        clip_fts_map = None
        if return_vis:
            # Use the penultimate CLIP token features and reshape token grid to feature map.
            clip_tokens = fts_all_stack[-2].permute(1, 0, 2)  # (B, L, C)
            clip_tokens = clip_tokens[:, 1:, :]  # drop cls token
            clip_h, clip_w = h // self.clip_flag, w // self.clip_flag
            expected_tokens = clip_h * clip_w
            if clip_tokens.shape[1] != expected_tokens:
                # Fallback for dynamic positional resize edge cases.
                clip_h = int(np.sqrt(clip_tokens.shape[1]))
                clip_w = clip_h
            clip_fts_map = clip_tokens.reshape(b, clip_h, clip_w, -1).permute(0, 3, 1, 2).contiguous()

        seg_sam3, seg_dino, fts, dino_fts = self._forward_seg_branches(img)
        _, _, fts_h, fts_w = fts.shape

        sam3_dino_fts = torch.cat([fts, dino_fts], dim=1)

        seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_sam3, dim=1)
        seg_dino_prob = seg_dino_prob.detach()

        attn_fts = F.interpolate(sam3_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]

            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

            require_seg_trans = True
            seg_dino_cam = seg_dino_prob[i]

            # iter_w = 0.5

            cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans,
                                                                   seg_dino_cam=seg_dino_cam,
                                                                   clip_flag = self.clip_flag
                                                                          )

            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            cams = cam_dict['refined_cam'].cuda()

            if return_vis:
                cam_score_map = torch.max(cams, dim=0, keepdim=False)[0]
                cam_score_list.append(cam_score_map.detach().cpu())

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

            cams = torch.cat([bg_score, cams], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        if self.training:
            return seg_sam3, seg_dino, all_cam_labels, attn_pred, fts, dino_fts
        else:
            if return_vis or return_grad_stats:
                vis_payload = {
                    'sam3_fts': fts if return_grad_stats else fts.detach(),
                    'dino_fts': dino_fts if return_grad_stats else dino_fts.detach(),
                    'seg_sam3': seg_sam3 if return_grad_stats else seg_sam3.detach(),
                    'seg_dino': seg_dino if return_grad_stats else seg_dino.detach(),
                    'attn_pred': attn_pred if return_grad_stats else attn_pred.detach(),
                    'clip_fts': clip_fts_map if return_grad_stats else (clip_fts_map.detach() if clip_fts_map is not None else None),
                    'cam_score_maps': torch.stack(cam_score_list, dim=0) if len(cam_score_list) > 0 else None,
                }
                return seg_sam3, seg_dino, all_cam_labels, attn_pred, vis_payload
            return seg_sam3, seg_dino, all_cam_labels, attn_pred

        
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    rsl = RSL('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    rsl._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    rsl(dummy_input)









