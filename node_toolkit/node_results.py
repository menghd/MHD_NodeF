"""
MHD_Nodet Project - Results Module
==================================
This module defines loss functions and evaluation metrics for the MHD_Nodet project, supporting regression, classification, and segmentation tasks.
- Includes losses: L-p loss, focal loss, Dice loss, and IoU loss.
- Includes metrics: MSE, accuracy, specificity, recall, precision, F1, Dice, and IoU.

项目：MHD_Nodet - 结果模块
本模块定义了 MHD_Nodet 项目的损失函数和评估指标，支持回归、分类和分割任务。
- 包含损失函数：L-p 损失、Focal 损失、Dice 损失和 IoU 损失。
- 包含评估指标：MSE、准确率、特异性、召回率、精确率、F1、Dice 和 IoU。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import torch
import logging

logger = logging.getLogger(__name__)

def validate_one_hot(tensor, num_classes):
    """
    Validate if tensor is approximately one-hot encoded.
    验证张量是否为近似的独热编码。
    """
    try:
        if tensor.shape[1] != num_classes:
            logger.error(f"Expected {num_classes} classes, got {tensor.shape[1]} channels")
            raise ValueError(f"target_tensor 的类别维度应等于 {num_classes}")
        # Check if tensor sums to 1 across class dimension
        if not torch.allclose(tensor.sum(dim=1), torch.ones_like(tensor[:, 0]), atol=1e-5):
            logger.warning("target_tensor 的通道和未接近于 1")
        # Check if values are in [0, 1]
        if not torch.all((tensor >= -1e-5) & (tensor <= 1 + 1e-5)):
            logger.warning("target_tensor 的值不在 [0, 1] 范围内")
    except Exception as e:
        logger.error(f"One-hot validation failed: {str(e)}")
        raise

def get_valid_classes(target_tensor, num_classes):
    """
    Identify classes with non-zero samples in target_tensor.
    确定目标数据中实际存在的类别（样本数大于0的类别）。
    """
    try:
        target_tensor = target_tensor.argmax(dim=1).flatten()
        class_counts = torch.bincount(target_tensor, minlength=num_classes)
        valid_classes = torch.where(class_counts > 0)[0]
        logger.debug(f"Valid classes: {valid_classes.tolist()}")
        return valid_classes
    except Exception as e:
        logger.error(f"Error in get_valid_classes: {str(e)}")
        return torch.tensor([], dtype=torch.int64, device=target_tensor.device)

def node_lp_loss(src_tensor, target_tensor, p=1.0):
    """
    L-p loss for regression tasks.
    回归任务的 L-p 损失。
    """
    try:
        src_tensor = src_tensor.contiguous().flatten()
        target_tensor = target_tensor.contiguous().flatten()
        if src_tensor.shape != target_tensor.shape:
            logger.error(f"Shape mismatch in node_lp_loss: src {src_tensor.shape}, target {target_tensor.shape}")
            raise ValueError("src_tensor and target_tensor must have the same shape")
        diff = torch.abs(src_tensor - target_tensor)
        loss = torch.pow(diff, p).mean()
        return loss
    except Exception as e:
        logger.error(f"Error in node_lp_loss: {str(e)}")
        return torch.tensor(0.0, device=src_tensor.device)

def node_focal_loss(src_tensor, target_tensor, alpha=None, gamma=2.0):
    """
    Focal loss for classification tasks, handling class imbalance.
    分类任务的 Focal 损失，处理类别不平衡。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        src_tensor = src_tensor.contiguous()
        target_tensor = target_tensor.contiguous().float()
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)

        # Clip probabilities for numerical stability
        pt = torch.clamp(src_tensor, min=1e-7, max=1-1e-7)
        logpt = torch.log(pt)
        loss = -target_tensor * (1 - pt) ** gamma * logpt

        if alpha is not None:
            alpha = torch.tensor(alpha, device=src_tensor.device)
            if alpha.shape[0] != num_classes:
                logger.error(f"Alpha shape {alpha.shape} does not match num_classes {num_classes}")
                raise ValueError(f"Alpha must have length {num_classes}")
            alpha = alpha.view(1, -1, *([1] * (src_tensor.dim() - 2))).expand_as(target_tensor)
            loss = loss * alpha

        # Compute loss only for valid classes
        if valid_classes.numel() > 0:
            loss = loss[:, valid_classes].mean()
        else:
            logger.warning("No valid classes found, returning zero loss")
            loss = torch.tensor(0.0, device=src_tensor.device)
        return loss
    except Exception as e:
        logger.error(f"Error in node_focal_loss: {str(e)}")
        return torch.tensor(0.0, device=src_tensor.device)

def node_dice_loss(src_tensor, target_tensor, smooth=1e-7):
    """
    Dice loss for segmentation tasks.
    分割任务的 Dice 损失。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        src_tensor = src_tensor.contiguous()
        target_tensor = target_tensor.contiguous().float()
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)

        spatial_dims = tuple(range(2, src_tensor.dim()))
        intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
        union = src_tensor.sum(dim=spatial_dims) + target_tensor.sum(dim=spatial_dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)

        if valid_classes.numel() > 0:
            dice = dice[valid_classes]
            return 1.0 - dice.mean()
        logger.warning("No valid classes found, returning zero loss")
        return torch.tensor(0.0, device=src_tensor.device)
    except Exception as e:
        logger.error(f"Error in node_dice_loss: {str(e)}")
        return torch.tensor(0.0, device=src_tensor.device)

def node_iou_loss(src_tensor, target_tensor, smooth=1e-7):
    """
    IoU loss for segmentation tasks.
    分割任务的 IoU 损失。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        src_tensor = src_tensor.contiguous()
        target_tensor = target_tensor.contiguous().float()
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)

        spatial_dims = tuple(range(2, src_tensor.dim()))
        intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
        union = (src_tensor + target_tensor - src_tensor * target_tensor).sum(dim=spatial_dims)
        iou = (intersection + smooth) / (union + smooth)

        if valid_classes.numel() > 0:
            iou = iou[valid_classes]
            return 1.0 - iou.mean()
        logger.warning("No valid classes found, returning zero loss")
        return torch.tensor(0.0, device=src_tensor.device)
    except Exception as e:
        logger.error(f"Error in node_iou_loss: {str(e)}")
        return torch.tensor(0.0, device=src_tensor.device)

def node_mse_metric(src_tensor, target_tensor):
    """
    Mean Squared Error metric for regression tasks.
    回归任务的均方误差指标。
    """
    try:
        src_tensor = src_tensor.contiguous().flatten()
        target_tensor = target_tensor.contiguous().flatten()
        if src_tensor.shape != target_tensor.shape:
            logger.error(f"Shape mismatch in node_mse_metric: src {src_tensor.shape}, target {target_tensor.shape}")
            raise ValueError("src_tensor and target_tensor must have the same shape")
        mse = torch.mean((src_tensor - target_tensor) ** 2).item()
        return {"per_class": [mse], "avg": mse}
    except Exception as e:
        logger.error(f"Error in node_mse_metric: {str(e)}")
        return {"per_class": [0.0], "avg": 0.0}

def node_accuracy_metric(src_tensor, target_tensor):
    """
    Accuracy metric for classification tasks.
    分类任务的准确率指标。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        src_tensor = src_tensor.argmax(dim=1).flatten()
        target_tensor = target_tensor.argmax(dim=1).flatten()
        
        correct = (src_tensor == target_tensor).float()
        overall_accuracy = correct.mean().item()
        
        per_class_accuracy = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            class_mask = (target_tensor == c)
            if class_mask.sum() > 0:
                per_class_accuracy[c] = correct[class_mask].mean().item()
        
        return {
            "per_class": per_class_accuracy.tolist(),
            "avg": per_class_accuracy[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0,
            "overall": overall_accuracy
        }
    except Exception as e:
        logger.error(f"Error in node_accuracy_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0, "overall": 0.0}

def node_specificity_metric(src_tensor, target_tensor):
    """
    Specificity metric for classification tasks.
    分类任务的特异性指标。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        src_tensor = src_tensor.argmax(dim=1).flatten()
        target_tensor = target_tensor.argmax(dim=1).flatten()
        
        hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
        hist = hist.reshape(num_classes, num_classes)
        
        TP = torch.diag(hist)
        FP = hist.sum(dim=0) - TP
        FN = hist.sum(dim=1) - TP
        TN = hist.sum() - (TP + FP + FN)
        
        specificity = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            specificity[c] = TN[c] / (TN[c] + FP[c] + 1e-7) if TN[c] + FP[c] > 0 else 0.0
        
        return {
            "per_class": specificity.tolist(),
            "avg": specificity[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Error in node_specificity_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0}

def node_recall_metric(src_tensor, target_tensor):
    """
    Recall metric for classification tasks.
    分类任务的召回率指标。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        src_tensor = src_tensor.argmax(dim=1).flatten()
        target_tensor = target_tensor.argmax(dim=1).flatten()
        
        hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
        hist = hist.reshape(num_classes, num_classes)
        
        TP = torch.diag(hist)
        FN = hist.sum(dim=1) - TP
        
        recall = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            recall[c] = TP[c] / (TP[c] + FN[c] + 1e-7) if TP[c] + FN[c] > 0 else 0.0
        
        return {
            "per_class": recall.tolist(),
            "avg": recall[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Error in node_recall_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0}

def node_precision_metric(src_tensor, target_tensor):
    """
    Precision metric for classification tasks.
    分类任务的精确率指标。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        src_tensor = src_tensor.argmax(dim=1).flatten()
        target_tensor = target_tensor.argmax(dim=1).flatten()
        
        hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
        hist = hist.reshape(num_classes, num_classes)
        
        TP = torch.diag(hist)
        FP = hist.sum(dim=0) - TP
        
        precision = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            precision[c] = TP[c] / (TP[c] + FP[c] + 1e-7) if TP[c] + FP[c] > 0 else 0.0
        
        return {
            "per_class": precision.tolist(),
            "avg": precision[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Error in node_precision_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0}

def node_f1_metric(src_tensor, target_tensor):
    """
    F1 score metric for classification tasks.
    分类任务的 F1 分数指标。
    """
    try:
        recall = node_recall_metric(src_tensor, target_tensor)["per_class"]
        precision = node_precision_metric(src_tensor, target_tensor)["per_class"]
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        f1 = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            p, r = precision[c], recall[c]
            f1[c] = 2 * p * r / (p + r + 1e-7) if p + r > 0 else 0.0
        
        return {
            "per_class": f1.tolist(),
            "avg": f1[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Error in node_f1_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0}

def node_dice_metric(src_tensor, target_tensor, smooth=1e-7):
    """
    Dice coefficient metric for segmentation tasks.
    分割任务的 Dice 系数指标。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        src_tensor = src_tensor.argmax(dim=1)
        target_tensor = target_tensor.argmax(dim=1)
        
        dice = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            pred_c = (src_tensor == c).float()
            target_c = (target_tensor == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice[c] = (2.0 * intersection + smooth) / (union + smooth) if union > 0 else 0.0
        
        return {
            "per_class": dice.tolist(),
            "avg": dice[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Error in node_dice_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0}

def node_iou_metric(src_tensor, target_tensor, smooth=1e-7):
    """
    IoU metric for segmentation tasks.
    分割任务的 IoU 指标。
    """
    try:
        validate_one_hot(target_tensor, src_tensor.shape[1])
        num_classes = src_tensor.shape[1]
        valid_classes = get_valid_classes(target_tensor, num_classes)
        
        src_tensor = src_tensor.argmax(dim=1)
        target_tensor = target_tensor.argmax(dim=1)
        
        iou = torch.full((num_classes,), 0.0, device=src_tensor.device)
        for c in range(num_classes):
            pred_c = (src_tensor == c).float()
            target_c = (target_tensor == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            iou[c] = (intersection + smooth) / (union + smooth) if union > 0 else 0.0
        
        return {
            "per_class": iou.tolist(),
            "avg": iou[valid_classes].mean().item() if valid_classes.numel() > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Error in node_iou_metric: {str(e)}")
        return {"per_class": [0.0] * src_tensor.shape[1], "avg": 0.0}
