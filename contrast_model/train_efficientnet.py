import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.amp
import nibabel as nib
import pandas as pd
import numpy as np
from pathlib import Path
import torchio as tio
import logging
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import csv
import uuid

# 设置日志
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集
class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cases = set(f.stem.split('_')[0] + '_' + f.stem.split('_')[1] for f in self.data_dir.glob('*.nii.gz'))
   
    def __len__(self):
        return len(self.cases)
   
    def __getitem__(self, idx):
        case_id = list(self.cases)[idx]
        images = []
        for i in range(5):
            img_path = self.data_dir / f"{case_id}_{i:04d}.nii.gz"
            img = nib.load(img_path).get_fdata()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append(img)
        images = np.stack(images, axis=0).astype(np.float32)
       
        label_path = self.data_dir / f"{case_id}_0006.csv"
        label = pd.read_csv(label_path)['Value'].iloc[0]
       
        images_tensor = torch.tensor(images, dtype=torch.float32)
        if self.transform:
            images_tensor = self.transform(images_tensor)
       
        return images_tensor, torch.tensor(label, dtype=torch.long)

# EfficientNet3D核心组件
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv3dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = Swish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcitation3D(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced_dim, 1, bias=True),
            Swish(),
            nn.Conv3d(reduced_dim, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.skip_connection = (stride == 1) and (in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        self.expand = nn.Identity() if expand_ratio == 1 else Conv3dBN(in_channels, hidden_dim, 1)
        self.depthwise = Conv3dBN(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim)
        
        reduced_dim = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation3D(hidden_dim, reduced_dim)
        
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.skip_connection:
            x += residual
        return x

# 3D EfficientNet-B0模型（适配5通道输入，4分类输出）
class EfficientNetB0_3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # EfficientNet-B0 3D配置
        config = [
            # (expand_ratio, channels, repeats, kernel_size, stride)
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        # 初始卷积层（适配5通道输入）
        self.stem = Conv3dBN(5, 32, 3, stride=2, padding=1)
        
        # 构建MBConv块
        layers = []
        in_channels = 32
        for expand_ratio, channels, repeats, kernel_size, stride in config:
            for _ in range(repeats):
                layers.append(MBConv3D(in_channels, channels, kernel_size, stride, expand_ratio))
                in_channels = channels
                stride = 1  # 仅第一个块用stride>1
        self.blocks = nn.Sequential(*layers)
        
        # 头部
        self.head = nn.Sequential(
            Conv3dBN(in_channels, 1280, 1),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(0.4),
            nn.Conv3d(1280, num_classes, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.view(x.size(0), -1)

# Poly学习率调度器
class PolyLRScheduler:
    def __init__(self, optimizer, base_lr, max_epochs, power=0.9):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.power = power
        self.current_epoch = 0
   
    def step(self):
        self.current_epoch += 1
        lr = self.base_lr * (1 - self.current_epoch / self.max_epochs) ** self.power
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 训练函数
def train_model(model, train_loader, val_loader, save_dir, num_epochs=100, patience=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = PolyLRScheduler(optimizer, base_lr=0.001, max_epochs=num_epochs)
    scaler = torch.amp.GradScaler('cuda')
   
    best_f1_weighted = 0.0
    early_stop_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    metrics_file = os.path.join(save_dir, 'metrics.csv')
   
    # 初始化CSV文件
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc',
                            'Precision_0', 'Precision_1', 'Precision_2', 'Precision_3',
                            'Recall_0', 'Recall_1', 'Recall_2', 'Recall_3',
                            'F1_0', 'F1_1', 'F1_2', 'F1_3',
                            'Precision_Macro', 'Recall_Macro', 'F1_Macro',
                            'Precision_Micro', 'Recall_Micro', 'F1_Micro',
                            'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted'])
   
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
       
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
           
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
           
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
           
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
       
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
       
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
       
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
       
        # 计算混淆矩阵和其他指标
        try:
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0)
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro', zero_division=0)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0)
        except Exception as e:
            logging.error(f"Error computing metrics: {str(e)}")
            cm = np.zeros((4, 4))
            precision = recall = f1 = [0.0] * 4
            precision_macro = recall_macro = f1_macro = 0.0
            precision_micro = recall_micro = f1_micro = 0.0
            precision_weighted = recall_weighted = f1_weighted = 0.0
       
        # 记录指标
        log_message = (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}\n'
                       f'Confusion Matrix:\n{cm}\n'
                       f'Precision: {precision.tolist()}\nRecall: {recall.tolist()}\nF1: {f1.tolist()}\n'
                       f'Macro: Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, F1={f1_macro:.4f}\n'
                       f'Micro: Precision={precision_micro:.4f}, Recall={recall_micro:.4f}, F1={f1_micro:.4f}\n'
                       f'Weighted: Precision={precision_weighted:.4f}, Recall={recall_weighted:.4f}, F1={f1_weighted:.4f}')
        logging.info(log_message)
        print(log_message)
       
        # 保存指标到CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, epoch_loss, train_acc, val_loss, val_acc] +
                           precision.tolist() + recall.tolist() + f1.tolist() +
                           [precision_macro, recall_macro, f1_macro,
                            precision_micro, recall_micro, f1_micro,
                            precision_weighted, recall_weighted, f1_weighted])
       
        # 学习率调度
        scheduler.step()
       
        # 保存最佳模型（基于加权F1）
        if f1_weighted > best_f1_weighted:
            best_f1_weighted = f1_weighted
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Saved best model with Weighted F1: {best_f1_weighted:.4f}')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logging.info(f'Early stopping at epoch {epoch+1}')
                print(f'Early stopping at epoch {epoch+1}')
                break

# 主函数
def main():
    train_transform = tio.Compose([
        tio.RandomFlip(axes=(0,1,2), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=10, p=0.5),
    ])
   
    train_dir = "/data/menghaoding/thu_xwh/TrainNiigzCsvData/Tr_fold1/train"
    val_dir = "/data/menghaoding/thu_xwh/TrainNiigzCsvData/Tr_fold1/val"
    save_dir = "/data/menghaoding/thu_xwh/f1_efficientnetb0_fold1"  # 修改保存目录标识
   
    setup_logging(save_dir)
   
    train_dataset = MRIDataset(train_dir, transform=train_transform)
    val_dataset = MRIDataset(val_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
   
    model = EfficientNetB0_3D(num_classes=4).to(device)  # 替换为EfficientNetB0_3D
   
    train_model(model, train_loader, val_loader, save_dir)

if __name__ == "__main__":
    main()
