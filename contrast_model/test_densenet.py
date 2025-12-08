import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(filename='inference_densenet121.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 设置单块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# DenseNet3D基本组件
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        
        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        
        if self.drop_rate > 0:
            new_features = nn.Dropout(self.drop_rate)(new_features)
        
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            layers.append(layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

# 3D DenseNet121模型（适配5通道输入，4分类输出）
class DenseNet121_3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, drop_rate=0, num_classes=4):
        super(DenseNet121_3D, self).__init__()

        # 初始卷积层（适配5通道输入）
        num_init_features = 64
        self.features = nn.Sequential(
            nn.Conv3d(5, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # 构建DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # 最终批量归一化
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # 分类器（保持与原ResNet34一致的Dropout=0.4）
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 推断函数
def inference():
    # 数据路径（修改为DenseNet对应的模型路径）
    data_dir = Path(r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\scratch_imagesTs")
    model_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\unicom20251018\f1_densenet121_fold5\best_model.pth"
    
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
    
    # 初始化DenseNet121_3D模型
    model = DenseNet121_3D(num_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded DenseNet121 model weights from {model_path}")
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
        
        # 保存预测结果（修改文件名标识为densenet121）
        output_path = data_dir / f"{case_id}_f1_densenet121_fold5_gt_5015.npy"
        try:
            np.save(output_path, probabilities[0])  # 保存 [4] 形状的预测向量
            logging.info(f"Saved prediction for {case_id} to {output_path}")
            print(f"Saved prediction for {case_id} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save prediction for {case_id} to {output_path}: {str(e)}")
            print(f"Error: Failed to save prediction for {case_id} to {output_path}: {str(e)}")

if __name__ == "__main__":
    inference()
