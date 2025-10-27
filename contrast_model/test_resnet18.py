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

# ResNet基本块
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# 3D ResNet18模型
class ResNet18_3D(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=4):
        super(ResNet18_3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 推断函数
def inference():
    # 数据路径
    data_dir = Path(r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\scratch_imagesTs")
    model_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\unicom20251018\f1_resnet18_fold5\best_model.pth"
    
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
    model = ResNet18_3D(num_classes=4).to(device)
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
        output_path = data_dir / f"{case_id}_f1_resnet18_fold5_gt_5015.npy"
        try:
            np.save(output_path, probabilities[0])  # 保存 [4] 形状的预测向量
            logging.info(f"Saved prediction for {case_id} to {output_path}")
            print(f"Saved prediction for {case_id} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save prediction for {case_id} to {output_path}: {str(e)}")
            print(f"Error: Failed to save prediction for {case_id} to {output_path}: {str(e)}")

if __name__ == "__main__":
    inference()
