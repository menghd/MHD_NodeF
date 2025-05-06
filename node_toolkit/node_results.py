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

def validate_one_hot(tensor, num_classes):
    if tensor.shape[1] != num_classes:
        raise ValueError("target_tensor 的类别维度应等于 num_classes")
    if not torch.all(tensor.sum(dim=1) == 1) or not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError("target_tensor 应为 one-hot 编码")

def get_valid_classes(target_tensor, num_classes):
    """确定目标数据中实际存在的类别（样本数大于0的类别）。"""
    target_tensor = target_tensor.argmax(dim=1).flatten()
    class_counts = torch.bincount(target_tensor, minlength=num_classes)
    valid_classes = torch.where(class_counts > 0)[0]
    return valid_classes

def node_lp_loss(src_tensor, target_tensor, p=1.0):
    src_tensor = src_tensor.contiguous().flatten()
    target_tensor = target_tensor.contiguous().flatten()
    diff = torch.abs(src_tensor - target_tensor)
    return torch.pow(diff, p).mean()

def node_focal_loss(src_tensor, target_tensor, alpha=None, gamma=2.0):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    src_tensor = src_tensor.contiguous()
    target_tensor = target_tensor.contiguous().float()
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)

    # 限制 pt 的范围以提高数值稳定性
    pt = torch.clamp(src_tensor, min=1e-7, max=1-1e-7)
    logpt = torch.log(pt)
    loss = -target_tensor * (1 - pt) ** gamma * logpt

    if alpha is not None:
        alpha = torch.tensor(alpha, device=src_tensor.device)
        alpha = alpha.view(1, -1, *([1] * (src_tensor.dim() - 2))).expand_as(target_tensor)
        loss = loss * alpha

    # 只对有效类别计算损失
    if valid_classes.numel() > 0:
        loss = loss[:, valid_classes].mean()
    else:
        loss = torch.tensor(0.0, device=src_tensor.device)
    return loss

def node_dice_loss(src_tensor, target_tensor, smooth=1e-7):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    src_tensor = src_tensor.contiguous()
    target_tensor = target_tensor.contiguous().float()
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)

    spatial_dims = tuple(range(2, src_tensor.dim()))
    intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
    union = src_tensor.sum(dim=spatial_dims) + target_tensor.sum(dim=spatial_dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # 只对有效类别计算 Dice 损失
    if valid_classes.numel() > 0:
        dice = dice[valid_classes]
        return 1.0 - dice.mean()
    return torch.tensor(0.0, device=src_tensor.device)

def node_iou_loss(src_tensor, target_tensor, smooth=1e-7):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    src_tensor = src_tensor.contiguous()
    target_tensor = target_tensor.contiguous().float()
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)

    spatial_dims = tuple(range(2, src_tensor.dim()))
    intersection = (src_tensor * target_tensor).sum(dim=spatial_dims)
    union = (src_tensor + target_tensor - src_tensor * target_tensor).sum(dim=spatial_dims)
    iou = (intersection + smooth) / (union + smooth)

    # 只对有效类别计算 IoU 损失
    if valid_classes.numel() > 0:
        iou = iou[valid_classes]
        return 1.0 - iou.mean()
    return torch.tensor(0.0, device=src_tensor.device)

def node_mse_metric(src_tensor, target_tensor):
    src_tensor = src_tensor.contiguous().flatten()
    target_tensor = target_tensor.contiguous().flatten()
    mse = torch.mean((src_tensor - target_tensor) ** 2).item()
    return {"per_class": [mse], "avg": mse}

def node_accuracy_metric(src_tensor, target_tensor):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)
    
    src_tensor = src_tensor.argmax(dim=1).flatten()
    target_tensor = target_tensor.argmax(dim=1).flatten()
    
    correct = (src_tensor == target_tensor).float()
    overall_accuracy = correct.mean().item()
    
    per_class_accuracy = torch.full((num_classes,), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        class_mask = (target_tensor == c)
        if class_mask.sum() > 0:
            per_class_accuracy[c] = correct[class_mask].mean().item()
    
    return {
        "per_class": per_class_accuracy.tolist(),
        "avg": per_class_accuracy[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan'),
        "overall": overall_accuracy
    }

def node_specificity_metric(src_tensor, target_tensor):
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
    
    specificity = torch.full((num_classes,), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        specificity[c] = TN[c] / (TN[c] + FP[c] + 1e-7) if TN[c] + FP[c] > 0 else float('nan')
    
    return {
        "per_class": specificity.tolist(),
        "avg": specificity[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan')
    }

def node_recall_metric(src_tensor, target_tensor):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)
    
    src_tensor = src_tensor.argmax(dim=1).flatten()
    target_tensor = target_tensor.argmax(dim=1).flatten()
    
    hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
    hist = hist.reshape(num_classes, num_classes)
    
    TP = torch.diag(hist)
    FN = hist.sum(dim=1) - TP
    
    recall = torch.full((num_classes,), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        recall[c] = TP[c] / (TP[c] + FN[c] + 1e-7) if TP[c] + FN[c] > 0 else float('nan')
    
    return {
        "per_class": recall.tolist(),
        "avg": recall[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan')
    }

def node_precision_metric(src_tensor, target_tensor):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)
    
    src_tensor = src_tensor.argmax(dim=1).flatten()
    target_tensor = target_tensor.argmax(dim=1).flatten()
    
    hist = torch.bincount(num_classes * target_tensor + src_tensor, minlength=num_classes**2)
    hist = hist.reshape(num_classes, num_classes)
    
    TP = torch.diag(hist)
    FP = hist.sum(dim=0) - TP
    
    precision = torch.full((num_classes,), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        precision[c] = TP[c] / (TP[c] + FP[c] + 1e-7) if TP[c] + FP[c] > 0 else float('nan')
    
    return {
        "per_class": precision.tolist(),
        "avg": precision[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan')
    }

def node_f1_metric(src_tensor, target_tensor):
    recall = node_recall_metric(src_tensor, target_tensor)["per_class"]
    precision = node_precision_metric(src_tensor, target_tensor)["per_class"]
    valid_classes = get_valid_classes(target_tensor, src_tensor.shape[1])
    
    f1 = torch.full((len(recall),), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        p, r = precision[c], recall[c]
        f1[c] = 2 * p * r / (p + r + 1e-7) if p + r > 0 else float('nan')
    
    return {
        "per_class": f1.tolist(),
        "avg": f1[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan')
    }

def node_dice_metric(src_tensor, target_tensor, smooth=1e-7):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)
    
    src_tensor = src_tensor.argmax(dim=1)
    target_tensor = target_tensor.argmax(dim=1)
    
    dice = torch.full((num_classes,), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        pred_c = (src_tensor == c).float()
        target_c = (target_tensor == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice[c] = (2.0 * intersection + smooth) / (union + smooth) if union > 0 else float('nan')
    
    return {
        "per_class": dice.tolist(),
        "avg": dice[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan')
    }

def node_iou_metric(src_tensor, target_tensor, smooth=1e-7):
    validate_one_hot(target_tensor, src_tensor.shape[1])
    num_classes = src_tensor.shape[1]
    valid_classes = get_valid_classes(target_tensor, num_classes)
    
    src_tensor = src_tensor.argmax(dim=1)
    target_tensor = target_tensor.argmax(dim=1)
    
    iou = torch.full((num_classes,), float('nan'), device=src_tensor.device)
    for c in valid_classes:
        pred_c = (src_tensor == c).float()
        target_c = (target_tensor == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou[c] = (intersection + smooth) / (union + smooth) if union > 0 else float('nan')
    
    return {
        "per_class": iou.tolist(),
        "avg": iou[valid_classes].mean().item() if valid_classes.numel() > 0 else float('nan')
    }
