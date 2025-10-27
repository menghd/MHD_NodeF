import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(filename='inference.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 设置单块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

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

# 推断函数
def inference():
    # 数据路径
    data_dir = Path(r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\scratch_imagesTs")
    model_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\unicom20251018\f1_inceptionv3_fold5\best_model.pth"
    
    # 检查数据路径是否存在
    if not data_dir.exists():
        logging.error(f"Data directory {data_dir} does not exist")
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # 检查模型文件是否存在
    if not Path(model_path).exists():
        logging.error(f"Model file {model_path} does not exist")
        print(f"Error: Model file {model_path} does not exist")
        return
    
    # 初始化模型
    model = InceptionV3_3D(num_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded model weights from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model weights: {str(e)}")
        print(f"Error: Failed to load model weights: {str(e)}")
        return
    model.eval()
    
    # 获取所有.nii.gz文件
    all_nii_files = list(data_dir.glob('case_*.nii.gz'))
    logging.info(f"Found {len(all_nii_files)} .nii.gz files: {[f.name for f in all_nii_files]}")
    print(f"Found {len(all_nii_files)} .nii.gz files: {[f.name for f in all_nii_files]}")
    
    # 提取case编号
    cases = set(f.stem.split('_')[0] + '_' + f.stem.split('_')[1] for f in all_nii_files)
    logging.info(f"Found {len(cases)} cases: {cases}")
    print(f"Found {len(cases)} cases: {cases}")
    
    if not cases:
        logging.warning("No valid cases found in the data directory")
        print("Warning: No valid cases found in the data directory")
        return
    
    # 推断每个case
    required_indices = ['0000', '0001', '0002', '0003', '1004']
    for case_id in cases:
        # 检查是否包含所有必要的文件
        images = []
        for idx in required_indices:
            img_path = data_dir / f"{case_id}_{idx}.nii.gz"
            if not img_path.exists():
                logging.warning(f"{img_path} does not exist, skipping {case_id}")
                print(f"Warning: {img_path} does not exist, skipping {case_id}")
                break
            try:
                img = nib.load(img_path).get_fdata()
                # 归一化到[0,1]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                images.append(img)
                logging.info(f"Loaded {img_path}")
            except Exception as e:
                logging.error(f"Failed to load {img_path}: {str(e)}")
                print(f"Error: Failed to load {img_path}: {str(e)}")
                break
        
        # 跳过无法加载完整5个图像的case
        if len(images) != 5:
            logging.warning(f"Incomplete images for {case_id}, expected 5, got {len(images)}")
            print(f"Warning: Incomplete images for {case_id}, expected 5, got {len(images)}")
            continue
            
        # 堆叠为5通道 [5, 64, 64, 64]
        images = np.stack(images, axis=0).astype(np.float32)
        images_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 5, 64, 64, 64]
        
        # 推断
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    output = model(images_tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()  # 四通道预测向量 [1, 4]
        except Exception as e:
            logging.error(f"Inference failed for {case_id}: {str(e)}")
            print(f"Error: Inference failed for {case_id}: {str(e)}")
            continue
        
        # 保存预测结果到case_xxxx_5015.npy
        output_path = data_dir / f"{case_id}_f1_inception_fold5_gt_5015.npy"
        try:
            np.save(output_path, probabilities[0])  # 保存 [4] 形状的预测向量
            logging.info(f"Saved prediction for {case_id} to {output_path}")
            print(f"Saved prediction for {case_id} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save prediction for {case_id} to {output_path}: {str(e)}")
            print(f"Error: Failed to save prediction for {case_id} to {output_path}: {str(e)}")

if __name__ == "__main__":
    inference()
