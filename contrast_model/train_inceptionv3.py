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

# Inception模块
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_ch):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm3d(ch1x1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm3d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch3x3),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm3d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm3d(ch5x5),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, pool_ch, kernel_size=1),
            nn.BatchNorm3d(pool_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

# 3D InceptionV3模型
class InceptionV3_3D(nn.Module):
    def __init__(self, num_classes=4):
        super(InceptionV3_3D, self).__init__()
        self.conv1 = nn.Conv3d(5, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv3d(64, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(192)
        
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool2(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

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
    
    train_dir = "/data/menghaoding/thu_xwh/TrainNiigzCsvData/Tr_fold5/train"
    val_dir = "/data/menghaoding/thu_xwh/TrainNiigzCsvData/Tr_fold5/val"
    save_dir = "/data/menghaoding/thu_xwh/f1_inceptionv3_fold5"
    
    setup_logging(save_dir)
    
    train_dataset = MRIDataset(train_dir, transform=train_transform)
    val_dataset = MRIDataset(val_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
    
    model = InceptionV3_3D(num_classes=4).to(device)
    
    train_model(model, train_loader, val_loader, save_dir)

if __name__ == "__main__":
    main()
