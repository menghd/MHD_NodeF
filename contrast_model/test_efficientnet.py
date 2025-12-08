import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(filename='inference_efficientnetb0.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 设置单块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

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
# 若需匹配ResNet34参数量，可切换为B4配置（下方注释）
class EfficientNetB0_3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # EfficientNet-B0 3D配置（参数量~5M）
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
        
        # ---------------------- 可选：EfficientNet-B4配置（参数量~19M，更接近ResNet34）----------------------
        # config = [
        #     (1, 24, 1, 3, 1),
        #     (6, 32, 2, 3, 2),
        #     (6, 56, 2, 5, 2),
        #     (6, 112, 3, 3, 2),
        #     (6, 192, 3, 5, 1),
        #     (6, 320, 4, 5, 2),
        #     (6, 512, 1, 3, 1),
        # ]
        # ---------------------------------------------------------------------------------------------------

        # 初始卷积层（适配5通道输入）
        self.stem = Conv3dBN(5, 32, 3, stride=2, padding=1)  # B0: 32通道 | B4: 48通道（若用B4需改为48）
        
        # 构建MBConv块
        layers = []
        in_channels = 32  # B0: 32 | B4: 48（若用B4需改为48）
        for expand_ratio, channels, repeats, kernel_size, stride in config:
            for _ in range(repeats):
                layers.append(MBConv3D(in_channels, channels, kernel_size, stride, expand_ratio))
                in_channels = channels
                stride = 1  # 仅第一个块用stride>1
        self.blocks = nn.Sequential(*layers)
        
        # 头部（保持与原ResNet34一致的Dropout=0.4）
        self.head = nn.Sequential(
            Conv3dBN(in_channels, 1280, 1),  # B0: 1280 | B4: 2560（若用B4需改为2560）
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(0.4),
            nn.Conv3d(1280, num_classes, 1)  # B0: 1280 | B4: 2560（若用B4需改为2560）
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.view(x.size(0), -1)

# 推断函数
def inference():
    # 数据路径（修改为EfficientNet对应的模型路径）
    data_dir = Path(r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\scratch_imagesTs")
    model_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\unicom20251018\f1_efficientnetb0_fold5\best_model.pth"
    
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
    
    # 初始化EfficientNetB0_3D模型
    model = EfficientNetB0_3D(num_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded EfficientNetB0 model weights from {model_path}")
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
        
        # 保存预测结果（修改文件名标识为efficientnetb0）
        output_path = data_dir / f"{case_id}_f1_efficientnetb0_fold5_gt_5015.npy"
        try:
            np.save(output_path, probabilities[0])  # 保存 [4] 形状的预测向量
            logging.info(f"Saved prediction for {case_id} to {output_path}")
            print(f"Saved prediction for {case_id} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save prediction for {case_id} to {output_path}: {str(e)}")
            print(f"Error: Failed to save prediction for {case_id} to {output_path}: {str(e)}")

if __name__ == "__main__":
    inference()
