"""
MHD_Nodet Project - Results Module
==================================
This module defines loss functions and evaluation metrics for the MHD_Nodet project,
supporting regression, classification, and segmentation tasks.

项目：MHD_Nodet - 结果模块
本模块定义了 MHD_Nodet 项目的损失函数和评估指标，支持回归、分类和分割任务。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def validate_one_hot(tensor, num_classes):
    """
    Validate if tensor is one-hot encoded.
    验证张量是否为 one-hot 编码。

    Args:
        tensor: Input tensor.
        num_classes: Number of classes.
    """
    if tensor.shape[1] != num_classes:
        raise ValueError("target_tensor 的类别维度应等于 num_classes")
    if not torch.all(tensor.sum(dim=1) == 1) or not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError("target_tensor 应为 one-hot 编码")

def node_lp_loss(src_tensor, target_tensor, p=1.0):
    """
    Compute Lp loss for regression tasks.
    计算回归任务的 Lp 损失。

    Args:
        src_tensor: Source tensor.
        target_tensor: Target tensor.
        p: Power of the loss.

    Returns:
        Lp loss value.
    """
    src_tensor = src_tensor.contiguous().flatten()
    target_tensor = target_tensor.contiguous().flatten()
    diff = torch.abs(src_tensor - target_tensor)
    return torch.pow(diff, p).mean()

def node_focal_loss(src_tensor, target_tensor, alpha=None, gamma=2.0):
    """
    Compute Focal Loss for multi-class classification tasks.
    计算多分类任务的 Focal Loss。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.
        alpha: Class weights, optional.
        gamma: Focusing parameter.

    Returns:
        Focal loss value.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    src_tensor = src_tensor.contiguous()
    target_tensor = target_tensor.contiguous().float()

    pt = src_tensor
    logpt = torch.log(pt + 1e-8)
    loss = -target_tensor * (1 - pt) ** gamma * logpt

    if alpha is not None:
        alpha = torch.tensor(alpha, device=src_tensor.device)
        alpha = alpha.view(1, -1, *([1] * (src_tensor.dim() - 2))).expand_as(target_tensor)
        loss = loss * alpha

    return loss.mean()

def node_dice_loss(src_tensor, target_tensor, smooth=1e-7):
    """
    Compute Dice Loss for segmentation tasks.
    计算分割任务的 Dice Loss。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice loss value.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    src_tensor = src_tensor.contiguous()
    target_tensor = target_tensor.contiguous().float()
    spatial_dims = tuple(range(2, src_tensor.dim()))
    intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
    union = src_tensor.sum(dim=spatial_dims) + target_tensor.sum(dim=spatial_dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def node_iou_loss(src_tensor, target_tensor, smooth=1e-7):
    """
    Compute IoU Loss for segmentation tasks.
    计算分割任务的 IoU Loss。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        IoU loss value.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    src_tensor = src_tensor.contiguous()
    target_tensor = target_tensor.contiguous().float()
    spatial_dims = tuple(range(2, src_tensor.dim()))
    intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
    union = (src_tensor + target_tensor - src_tensor * target_tensor).sum(dim=spatial_dims)
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou.mean()

def node_mse_metric(src_tensor, target_tensor):
    """
    Compute MSE metric for regression tasks.
    计算回归任务的 MSE 指标。

    Args:
        src_tensor: Source tensor.
        target_tensor: Target tensor.

    Returns:
        Dictionary with per-class and average MSE.
    """
    src_tensor = src_tensor.contiguous().flatten()
    target_tensor = target_tensor.contiguous().flatten()
    mse = torch.mean((src_tensor - target_tensor) ** 2).item()
    return {"per_class": [mse], "avg": mse}

def node_recall_metric(src_tensor, target_tensor):
    """
    Compute Recall metric for multi-class tasks.
    计算多分类任务的 Recall 指标。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.

    Returns:
        Dictionary with per-class and average recall.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    src_tensor = src_tensor.argmax(dim=1).flatten()
    target_tensor = target_tensor.argmax(dim=1).flatten()

    hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
    hist = hist.reshape(num_classes, num_classes)

    TP = torch.diag(hist)
    FN = hist.sum(dim=1) - TP
    recall = torch.zeros(num_classes, device=src_tensor.device)
    for i in range(num_classes):
        recall[i] = TP[i] / (TP[i] + FN[i] + 1e-7) if TP[i] + FN[i] > 0 else 0.0

    return {"per_class": recall.tolist(), "avg": recall[TP + FN > 0].mean().item() if (TP + FN > 0).any() else 0.0}

def node_precision_metric(src_tensor, target_tensor):
    """
    Compute Precision metric for multi-class tasks.
    计算多分类任务的 Precision 指标。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.

    Returns:
        Dictionary with per-class and average precision.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    src_tensor = src_tensor.argmax(dim=1).flatten()
    target_tensor = target_tensor.argmax(dim=1).flatten()

    hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
    hist = hist.reshape(num_classes, num_classes)

    TP = torch.diag(hist)
    FP = hist.sum(dim=0) - TP
    precision = torch.zeros(num_classes, device=src_tensor.device)
    for i in range(num_classes):
        precision[i] = TP[i] / (TP[i] + FP[i] + 1e-7) if TP[i] + FP[i] > 0 else 0.0

    return {"per_class": precision.tolist(), "avg": precision[TP + FP > 0].mean().item() if (TP + FP > 0).any() else 0.0}

def node_f1_metric(src_tensor, target_tensor):
    """
    Compute F1 metric based on recall and precision.
    基于 recall 和 precision 计算 F1 指标。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.

    Returns:
        Dictionary with per-class and average F1 score.
    """
    recall = node_recall_metric(src_tensor, target_tensor)["per_class"]
    precision = node_precision_metric(src_tensor, target_tensor)["per_class"]
    f1 = [2 * p * r / (p + r + 1e-7) if p + r > 0 else 0.0 for p, r in zip(precision, recall)]
    return {"per_class": f1, "avg": np.mean(f1)}

def node_dice_metric(src_tensor, target_tensor, smooth=1e-7):
    """
    Compute Dice metric for segmentation tasks.
    计算分割任务的 Dice 指标。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dictionary with per-class and average Dice score.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    src_tensor = src_tensor.argmax(dim=1)
    target_tensor = target_tensor.argmax(dim=1)

    dice = torch.zeros(num_classes, device=src_tensor.device)
    for c in range(num_classes):
        pred_c = (src_tensor == c).float()
        target_c = (target_tensor == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice[c] = (2.0 * intersection + smooth) / (union + smooth) if union > 0 else 0.0

    return {"per_class": dice.tolist(), "avg": dice.mean().item()}

def node_iou_metric(src_tensor, target_tensor, smooth=1e-7):
    """
    Compute IoU metric for segmentation tasks.
    计算分割任务的 IoU 指标。

    Args:
        src_tensor: Source tensor [batch_size, C, *S], model output (after softmax).
        target_tensor: Target tensor [batch_size, C, *S], one-hot encoded.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dictionary with per-class and average IoU score.
    """
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    src_tensor = src_tensor.argmax(dim=1)
    target_tensor = target_tensor.argmax(dim=1)

    iou = torch.zeros(num_classes, device=src_tensor.device)
    for c in range(num_classes):
        pred_c = (src_tensor == c).float()
        target_c = (target_tensor == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou[c] = (intersection + smooth) / (union + smooth) if union > 0 else 0.0

    return {"per_class": iou.tolist(), "avg": iou.mean().item()}
