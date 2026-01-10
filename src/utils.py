import torch

def dice_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

    return dice.item()


def iou_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.item()
