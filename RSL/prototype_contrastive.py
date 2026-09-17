import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveLoss(nn.Module):
    """Class-Aware Prototype Contrastive Learning.

    Maintains momentum-updated class prototypes for SAM3 and DINO branches
    and computes:
      1) Per-branch pixel-to-prototype InfoNCE loss (high-confidence pixels only)
      2) Cross-branch prototype alignment loss (cosine similarity)
    """

    def __init__(self, num_classes, embedding_dim, momentum=0.999,
                 temperature=0.1, confidence_thresh=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        self.temperature = temperature
        self.confidence_thresh = confidence_thresh

        self.register_buffer('sam3_prototypes', torch.zeros(num_classes, embedding_dim))
        self.register_buffer('dino_prototypes', torch.zeros(num_classes, embedding_dim))
        self.register_buffer('sam3_proto_init', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('dino_proto_init', torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Prototype update (no grad)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_prototypes(self, fts, labels, confidence, proto_bank, init_flags):
        """EMA-update prototypes from high-confidence pixels.

        Args:
            fts:         [B, C, H, W]  feature maps (already at feature-map resolution)
            labels:      [B, H, W]     pseudo labels resized to feature-map resolution
            confidence:  [B, H, W]     confidence map resized to feature-map resolution
            proto_bank:  [K, C]        prototype memory bank
            init_flags:  [K]           whether each class slot has been initialised
        """
        B, C, H, W = fts.shape
        fts_flat = fts.permute(0, 2, 3, 1).reshape(-1, C)       # [N, C]
        labels_flat = labels.reshape(-1)                          # [N]
        conf_flat = confidence.reshape(-1)                        # [N]

        high_conf = conf_flat > self.confidence_thresh

        for cls_id in range(self.num_classes):
            mask = (labels_flat == cls_id) & high_conf
            if mask.sum() == 0:
                continue

            cls_fts = fts_flat[mask]                              # [M, C]
            cls_w = conf_flat[mask]
            cls_w = cls_w / cls_w.sum()

            new_proto = (cls_fts * cls_w.unsqueeze(1)).sum(dim=0)
            new_proto = F.normalize(new_proto, dim=0)

            if init_flags[cls_id]:
                proto_bank[cls_id] = (self.momentum * proto_bank[cls_id]
                                      + (1.0 - self.momentum) * new_proto)
            else:
                proto_bank[cls_id] = new_proto
                init_flags[cls_id] = True

            proto_bank[cls_id] = F.normalize(proto_bank[cls_id], dim=0)

    # ------------------------------------------------------------------
    # Per-branch pixel-to-prototype contrastive loss (InfoNCE)
    # ------------------------------------------------------------------
    def _pixel_proto_contrastive(self, fts, labels, confidence, proto_bank, init_flags):
        B, C, H, W = fts.shape

        fts_norm = F.normalize(fts, dim=1)
        fts_flat = fts_norm.permute(0, 2, 3, 1).reshape(-1, C)   # [N, C]
        labels_flat = labels.reshape(-1)
        conf_flat = confidence.reshape(-1)

        valid = ((labels_flat != 255)
                 & (conf_flat > self.confidence_thresh)
                 & (labels_flat < self.num_classes))

        active_classes = torch.unique(labels_flat[valid])
        proto_ready = init_flags[active_classes.long()].all() if active_classes.numel() > 0 else False
        if valid.sum() == 0 or not proto_ready:
            return fts.new_tensor(0.0)

        valid_fts = fts_flat[valid]          # [M, C]
        valid_labels = labels_flat[valid]    # [M]

        proto = F.normalize(proto_bank.clone(), dim=1)            # [K, C]
        logits = torch.mm(valid_fts, proto.t()) / self.temperature  # [M, K]

        loss = F.cross_entropy(logits, valid_labels.long())
        return loss

    # ------------------------------------------------------------------
    # Cross-branch prototype alignment
    # ------------------------------------------------------------------
    def _cross_branch_alignment(self):
        both_init = self.sam3_proto_init & self.dino_proto_init
        if both_init.sum() == 0:
            return self.sam3_prototypes.new_tensor(0.0)

        s_p = F.normalize(self.sam3_prototypes[both_init], dim=1)
        d_p = F.normalize(self.dino_prototypes[both_init], dim=1)
        loss = (1.0 - (s_p * d_p).sum(dim=1)).mean()
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, sam3_fts, dino_fts, pseudo_labels, confidence):
        """
        Args:
            sam3_fts:      [B, C, Hf, Wf]  SAM3 decoder embedding
            dino_fts:      [B, C, Hf, Wf]  DINO decoder embedding
            pseudo_labels: [B, H, W]        pseudo labels (original resolution)
            confidence:    [B, H, W]        pixel-level confidence in [0, 1]

        Returns:
            proto_loss      : scalar  weighted sum of all sub-losses
            loss_sam3       : scalar  SAM3 pixel-to-proto contrastive
            loss_dino       : scalar  DINO pixel-to-proto contrastive
            loss_align      : scalar  cross-branch prototype alignment
        """
        _, _, fts_h, fts_w = sam3_fts.shape

        labels = F.interpolate(
            pseudo_labels.unsqueeze(1).float(),
            size=(fts_h, fts_w), mode='nearest'
        ).squeeze(1).long()

        conf = F.interpolate(
            confidence.unsqueeze(1),
            size=(fts_h, fts_w), mode='bilinear', align_corners=False
        ).squeeze(1)

        # --- momentum prototype update (detached features) ---
        self._update_prototypes(sam3_fts.detach(), labels, conf,
                                self.sam3_prototypes, self.sam3_proto_init)
        self._update_prototypes(dino_fts.detach(), labels, conf,
                                self.dino_prototypes, self.dino_proto_init)

        # --- contrastive losses (gradients flow through features) ---
        loss_sam3 = self._pixel_proto_contrastive(
            sam3_fts, labels, conf, self.sam3_prototypes, self.sam3_proto_init)
        loss_dino = self._pixel_proto_contrastive(
            dino_fts, labels, conf, self.dino_prototypes, self.dino_proto_init)
        loss_align = self._cross_branch_alignment()

        proto_loss = 0.5 * (loss_sam3 + loss_dino) + 0.5 * loss_align

        return proto_loss, loss_sam3, loss_dino, loss_align
