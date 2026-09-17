import torch
import torch.nn.functional as F

def dice_loss(output, target, eps=1e-7):
    eps = 1e-7
    ignore_mask = target.eq(255)
    target_safe = target.clone()
    target_safe[ignore_mask] = output.shape[1]

    # convert target to onehot (extra channel for ignore index)
    targ_onehot = F.one_hot(target_safe.long(), num_classes=output.shape[1] + 1).permute(0, 3, 1, 2).float()
    targ_onehot = targ_onehot[:, :-1, :, :]

    # convert logits to probs
    pred = F.softmax(output, dim=1)

    valid_region = (~ignore_mask).unsqueeze(1).float()
    # sum over HW
    inter = (pred * valid_region * targ_onehot).sum(axis=[0, 2, 3])
    union = ((pred + targ_onehot) * valid_region).sum(axis=[0, 2, 3])
    # mean over C
    dice = (2. * inter / (union + eps)).mean()
    return 1. - dice
    

class DiceLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, output, targ):
        """
        output is NCHW, targ is NHW
        """
        return dice_loss(output, targ)

    def activation(self, output):
        return F.softmax(output, dim=1)
    
    def decodes(self, output):
        return output.argmax(1)