from torch import Tensor, argmax


def dice_score(logits: Tensor,
               targets: Tensor,
               eps: float = 1e-6) -> Tensor:
    preds = argmax(logits, dim=1)

    preds = preds.float()
    targets = targets.float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()
