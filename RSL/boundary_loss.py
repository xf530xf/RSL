import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeDetector(nn.Module):
    """Fixed-weight Sobel filters for differentiable edge detection."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # (2, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Edge magnitude per channel: (B, C, H, W)
        """
        b, c, h, w = x.shape
        x_flat = x.reshape(b * c, 1, h, w)
        edges = F.conv2d(x_flat, self.kernel, padding=1)  # (B*C, 2, H, W)
        mag = torch.sqrt(edges[:, 0:1] ** 2 + edges[:, 1:2] ** 2 + 1e-8)
        return mag.reshape(b, c, h, w)


def extract_label_boundary(label, ignore_index=255, dilation=1):
    """Extract binary boundary mask from an integer label map.

    A pixel is on the boundary if any of its (dilated) 4-connected neighbors
    carries a different valid class label.

    Args:
        label:        (B, H, W) integer labels
        ignore_index: value to treat as invalid
        dilation:     dilation of the boundary band (1 = 1-pixel boundary)

    Returns:
        boundary: (B, 1, H, W) float mask in {0, 1}
    """
    label_f = label.unsqueeze(1).float()  # (B, 1, H, W)
    pad = dilation
    padded = F.pad(label_f, [pad] * 4, mode='replicate')

    shifts = [
        padded[:, :, pad:pad + label.shape[1], :-2 * pad or None],   # left
        padded[:, :, pad:pad + label.shape[1], 2 * pad:],            # right
        padded[:, :, :-2 * pad or None, pad:pad + label.shape[2]],   # top
        padded[:, :, 2 * pad:, pad:pad + label.shape[2]],            # bottom
    ]

    diff = torch.zeros_like(label_f)
    for s in shifts:
        if s.shape != label_f.shape:
            s = s[:, :, :label_f.shape[2], :label_f.shape[3]]
        diff = diff + (s != label_f).float()

    boundary = (diff > 0).float()

    valid = (label != ignore_index).unsqueeze(1).float()
    return boundary * valid


class BoundaryAwareDistillationLoss(nn.Module):
    """Boundary-aware Distillation for weakly-supervised semantic segmentation.

    Two sub-losses:

    1. **Boundary Mutual KL Distillation** (`L_bkl`):
       Symmetric KL between the SAM3 and DINO seg branches, weighted by a
       boundary attention map so that the gradient signal is concentrated on
       the hard edge pixels where weak-supervision errors are most likely.

    2. **PAR-aware Boundary Fidelity** (`L_bfid`):
       Penalises segmentation edges that are *not* supported by image-level
       gradients.  This exploits the same pixel-affinity philosophy as PAR:
       if the image shows no intensity change, the segmentation should not
       show a class boundary either.
    """

    def __init__(self, temperature=2.0, boundary_dilation=2,
                 kl_weight=1.0, fidelity_weight=1.0):
        super().__init__()
        self.sobel = SobelEdgeDetector()
        self.temperature = temperature
        self.boundary_dilation = boundary_dilation
        self.kl_weight = kl_weight
        self.fidelity_weight = fidelity_weight

    def _seg_edge_map(self, logits):
        """Differentiable segmentation edge from the max-class probability."""
        prob = F.softmax(logits / self.temperature, dim=1)
        max_prob = prob.max(dim=1, keepdim=True)[0]    # (B, 1, H, W)
        return self.sobel(max_prob)                     # (B, 1, H, W)

    def _image_edge_map(self, img):
        """Normalised gray-scale edge magnitude from the input image."""
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        edge = self.sobel(gray)
        return edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-8)

    def forward(self, seg_sam3, seg_dino, img,
                pseudo_label=None, ignore_index=255):
        """
        Args:
            seg_sam3     : (B, C, H, W) logits -- SAM3 branch
            seg_dino     : (B, C, H, W) logits -- DINO branch
            img          : (B, 3, H', W') input image (any resolution)
            pseudo_label : (B, H, W) pseudo ground-truth labels
            ignore_index : label value to mask out

        Returns:
            total_loss, bkl_loss, bfid_loss
        """
        _, _, h, w = seg_sam3.shape

        # ---- boundary attention mask from pseudo-label ----
        if pseudo_label is not None:
            boundary_mask = extract_label_boundary(
                pseudo_label, ignore_index, dilation=self.boundary_dilation
            )  # (B, 1, H, W)
            valid = (pseudo_label != ignore_index).unsqueeze(1).float()
        else:
            boundary_mask = torch.ones(
                seg_sam3.shape[0], 1, h, w, device=seg_sam3.device
            )
            valid = boundary_mask

        # ======================================================
        # 1.  Boundary Mutual KL Distillation  (L_bkl)
        # ======================================================
        log_p_sam3 = F.log_softmax(seg_sam3 / self.temperature, dim=1)
        log_p_dino = F.log_softmax(seg_dino / self.temperature, dim=1)
        p_sam3 = F.softmax(seg_sam3 / self.temperature, dim=1)
        p_dino = F.softmax(seg_dino / self.temperature, dim=1)

        kl_sd = (p_sam3 * (log_p_sam3 - log_p_dino)).sum(dim=1, keepdim=True)
        kl_ds = (p_dino * (log_p_dino - log_p_sam3)).sum(dim=1, keepdim=True)
        sym_kl = 0.5 * (kl_sd + kl_ds)  # (B, 1, H, W)

        weight = boundary_mask * valid
        bkl_loss = (sym_kl * weight).sum() / (weight.sum() + 1e-8)
        bkl_loss = bkl_loss * (self.temperature ** 2)

        # ======================================================
        # 2.  PAR-aware Boundary Fidelity  (L_bfid)
        # ======================================================
        img_resized = F.interpolate(
            img, size=(h, w), mode='bilinear', align_corners=False
        )
        img_edge = self._image_edge_map(img_resized)      # (B, 1, H, W)

        fused_logits = 0.5 * seg_sam3 + 0.5 * seg_dino
        seg_edge = self._seg_edge_map(fused_logits)        # (B, 1, H, W)
        seg_edge = seg_edge / (seg_edge.amax(dim=(2, 3), keepdim=True) + 1e-8)

        bfid_loss = F.mse_loss(
            seg_edge * valid, img_edge.detach() * valid, reduction='sum'
        ) / (valid.sum() + 1e-8)

        total = self.kl_weight * bkl_loss + self.fidelity_weight * bfid_loss
        return total, bkl_loss, bfid_loss
