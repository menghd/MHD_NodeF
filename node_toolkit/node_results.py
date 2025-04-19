import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

def node_lp_loss(src_tensor, target_tensor, p=1.0):
    """计算Lp损失，用于回归任务"""
    src_tensor = src_tensor.contiguous().flatten()
    target_tensor = target_tensor.contiguous().flatten()
    diff = torch.abs(src_tensor - target_tensor)
    return torch.pow(diff, p).mean()

def node_focal_loss(src_tensor, target_tensor, alpha=None, gamma=2.0):
    """
    计算Focal Loss，用于分类任务
    src_tensor: [batch_size, C, *S], 模型输出（未经过sigmoid）
    target_tensor: [batch_size, C, *S], 已为one-hot编码（分类）或float（分割）
    """
    src_tensor = src_tensor.contiguous()  # [batch_size, C, *S]
    target_tensor = target_tensor.contiguous()  # [batch_size, C, *S]

    # 应用sigmoid
    pt = torch.sigmoid(src_tensor)  # [batch_size, C, *S]

    # 确保target_tensor是float类型，假设已在NodeDataset中完成one-hot编码
    target_tensor = target_tensor.float()

    # 计算focal loss
    logpt = torch.log(pt + 1e-8)
    logpt_neg = torch.log(1 - pt + 1e-8)

    loss = -target_tensor * (1 - pt) ** gamma * logpt
    loss += -(1 - target_tensor) * pt ** gamma * logpt_neg

    if alpha is not None:
        alpha = torch.tensor(alpha, device=src_tensor.device)
        alpha = alpha.view(1, -1, *([1] * (src_tensor.dim() - 2))).expand_as(target_tensor)
        loss = loss * alpha

    return loss.mean()

def node_dice_loss(src_tensor, target_tensor, smooth=1e-7):
    """计算Dice Loss，用于分割任务"""
    src_tensor = torch.sigmoid(src_tensor)  # [batch_size, C, *S]
    target_tensor = target_tensor.float()  # [batch_size, C, *S]
    spatial_dims = tuple(range(2, src_tensor.dim()))
    intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
    union = src_tensor.sum(dim=spatial_dims) + target_tensor.sum(dim=spatial_dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def node_iou_loss(src_tensor, target_tensor, smooth=1e-7):
    """计算IoU Loss，用于分割任务"""
    src_tensor = torch.sigmoid(src_tensor)  # [batch_size, C, *S]
    target_tensor = target_tensor.float()  # [batch_size, C, *S]
    spatial_dims = tuple(range(2, src_tensor.dim()))
    intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
    union = (src_tensor + target_tensor - src_tensor * target_tensor).sum(dim=spatial_dims)
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou.mean()

def node_mse_metric(src_tensor, target_tensor):
    """计算MSE指标，用于回归任务"""
    src_tensor = src_tensor.contiguous().flatten()
    target_tensor = target_tensor.contiguous().flatten()
    mse = torch.mean((src_tensor - target_tensor) ** 2).item()
    return {"per_class": [mse], "avg": mse}

def node_recall_metric(src_tensor, target_tensor):
    """
    计算Recall指标，用于分类和分割任务
    src_tensor: [batch_size, C, *S], 模型输出（未经过sigmoid）
    target_tensor: [batch_size, C, *S], 已为one-hot编码
    """
    num_classes = src_tensor.shape[1]
    src_tensor = torch.sigmoid(src_tensor) > 0.5  # [batch_size, C, *S]
    src_tensor = src_tensor.long().flatten().cpu().numpy()
    target_tensor = target_tensor.long().flatten().cpu().numpy()

    unique_labels = np.unique(target_tensor)
    if len(unique_labels) == 0:
        recall = np.full(num_classes, np.nan)
        return {"per_class": recall.tolist(), "avg": np.nanmean(recall)}

    labels = list(range(num_classes))
    try:
        cm = confusion_matrix(target_tensor, src_tensor, labels=labels)
        TP = np.diag(cm)
        FN = cm.sum(axis=1) - TP
        recall = np.zeros(num_classes)
        for i in range(num_classes):
            if i in unique_labels:
                recall[i] = TP[i] / (TP[i] + FN[i] + 1e-7)
            else:
                recall[i] = np.nan
    except ValueError:
        recall = np.full(num_classes, np.nan)
        for i in unique_labels:
            if i < num_classes:
                sub_cm = confusion_matrix(target_tensor, src_tensor, labels=[i])
                TP = sub_cm[0, 0]
                FN = sub_cm[0, 1]
                recall[i] = TP / (TP + FN + 1e-7)

    return {"per_class": recall.tolist(), "avg": np.nanmean(recall)}

def node_precision_metric(src_tensor, target_tensor):
    """
    计算Precision指标，用于分类和分割任务
    src_tensor: [batch_size, C, *S], 模型输出（未经过sigmoid）
    target_tensor: [batch_size, C, *S], 已为one-hot编码
    """
    num_classes = src_tensor.shape[1]
    src_tensor = torch.sigmoid(src_tensor) > 0.5  # [batch_size, C, *S]
    src_tensor = src_tensor.long().flatten().cpu().numpy()
    target_tensor = target_tensor.long().flatten().cpu().numpy()

    unique_labels = np.unique(target_tensor)
    if len(unique_labels) == 0:
        precision = np.full(num_classes, np.nan)
        return {"per_class": precision.tolist(), "avg": np.nanmean(precision)}

    labels = list(range(num_classes))
    try:
        cm = confusion_matrix(target_tensor, src_tensor, labels=labels)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        precision = np.zeros(num_classes)
        for i in range(num_classes):
            if i in unique_labels:
                precision[i] = TP[i] / (TP[i] + FP[i] + 1e-7)
            else:
                precision[i] = np.nan
    except ValueError:
        precision = np.full(num_classes, np.nan)
        for i in unique_labels:
            if i < num_classes:
                sub_cm = confusion_matrix(target_tensor, src_tensor, labels=[i])
                TP = sub_cm[0, 0]
                FP = sub_cm[1, 0]
                precision[i] = TP / (TP + FP + 1e-7)

    return {"per_class": precision.tolist(), "avg": np.nanmean(precision)}

def node_f1_metric(src_tensor, target_tensor):
    """计算F1指标，基于recall和precision"""
    recall = node_recall_metric(src_tensor, target_tensor)["per_class"]
    precision = node_precision_metric(src_tensor, target_tensor)["per_class"]
    f1 = [2 * p * r / (p + r + 1e-7) if not (np.isnan(p) or np.isnan(r)) else np.nan for p, r in zip(precision, recall)]
    return {"per_class": f1, "avg": np.nanmean(f1)}

def node_dice_metric(src_tensor, target_tensor):
    """
    计算Dice指标，用于分割任务
    src_tensor: [batch_size, C, *S], 模型输出（未经过sigmoid）
    target_tensor: [batch_size, C, *S], 已为one-hot编码
    """
    num_classes = src_tensor.shape[1]
    src_tensor = torch.sigmoid(src_tensor) > 0.5  # [batch_size, C, *S]
    src_tensor = src_tensor.cpu().numpy()
    target_tensor = target_tensor.cpu().numpy()

    unique_labels = np.unique(target_tensor)
    if len(unique_labels) == 0:
        dice = np.full(num_classes, np.nan)
        return {"per_class": dice.tolist(), "avg": np.nanmean(dice)}

    dice = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (src_tensor == c).astype(np.float32)
        target_c = (target_tensor == c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if c in unique_labels:
            dice[c] = (2.0 * intersection + 1e-7) / (union + 1e-7)
        else:
            dice[c] = np.nan

    return {"per_class": dice.tolist(), "avg": np.nanmean(dice)}

def node_iou_metric(src_tensor, target_tensor):
    """
    计算IoU指标，用于分割任务
    src_tensor: [batch_size, C, *S], 模型输出（未经过sigmoid）
    target_tensor: [batch_size, C, *S], 已为one-hot编码
    """
    num_classes = src_tensor.shape[1]
    src_tensor = torch.sigmoid(src_tensor) > 0.5  # [batch_size, C, *S]
    src_tensor = src_tensor.cpu().numpy()
    target_tensor = target_tensor.cpu().numpy()

    unique_labels = np.unique(target_tensor)
    if len(unique_labels) == 0:
        iou = np.full(num_classes, np.nan)
        return {"per_class": iou.tolist(), "avg": np.nanmean(iou)}

    iou = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (src_tensor == c).astype(np.float32)
        target_c = (target_tensor == c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        if c in unique_labels:
            iou[c] = (intersection + 1e-7) / (union + 1e-7)
        else:
            iou[c] = np.nan

    return {"per_class": iou.tolist(), "avg": np.nanmean(iou)}
