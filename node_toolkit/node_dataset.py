import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F

class NodeDataset(Dataset):
    def __init__(self, data_dir, node_id, suffix, target_shape, transforms=None, num_dimensions=3, inference_mode=False):
        self.data_dir = data_dir
        self.node_id = node_id
        self.suffix = suffix
        self.target_shape = target_shape  # (channels, spatial_dims...)
        self.transforms = transforms or []
        self.num_dimensions = num_dimensions
        self.inference_mode = inference_mode

        # 获取所有文件
        all_files = sorted(os.listdir(data_dir))
        self.case_ids = sorted(set(f.split('_')[1] for f in all_files if f.startswith('case_')))

        # 确定文件扩展名
        file_ext = None
        for case_id in self.case_ids:
            data_file = f'case_{case_id}_{self.suffix}.nii.gz'
            if data_file in all_files:
                file_ext = '.nii.gz'
                break
            data_file = f'case_{case_id}_{self.suffix}.csv'
            if data_file in all_files:
                file_ext = '.csv'
                break
        if file_ext is None:
            raise ValueError(f"No valid files found for suffix {self.suffix} in {data_dir}")
        self.file_ext = file_ext

        # 验证文件存在性
        self.case_ids = [case_id for case_id in self.case_ids
                         if f'case_{case_id}_{self.suffix}{self.file_ext}' in all_files]
        if not self.case_ids:
            raise ValueError(f"No valid case IDs found for suffix {self.suffix}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        data_path = os.path.join(self.data_dir, f'case_{case_id}_{self.suffix}{self.file_ext}')

        # 加载数据
        if self.file_ext == '.nii.gz':
            data = nib.load(data_path).get_fdata()
            data_array = np.asarray(data, dtype=np.float32)
            # 移除可能的额外维度
            data_array = np.squeeze(data_array)
            # 确保至少是 num_dimensions 维
            while len(data_array.shape) < self.num_dimensions:
                data_array = np.expand_dims(data_array, axis=0)
            # 调整为 [C, *S]，假设输入是 [H, W, D]，添加 channel 维度
            if len(data_array.shape) == self.num_dimensions:
                data_array = np.expand_dims(data_array, axis=0)  # [1, H, W, D]
        else:  # .csv
            df = pd.read_csv(data_path)
            if 'Value' not in df.columns:
                raise ValueError(f"CSV file {data_path} does not have 'Value' column")
            value = df['Value'].iloc[0]
            if isinstance(value, (int, float)):
                # 标量值，扩展到目标空间维度
                data_array = np.full([1] + list(self.target_shape[1:]), float(value), dtype=np.float32)
            else:
                # 向量值
                value = np.array(value, dtype=np.float32)
                data_array = value
                # 移除可能的额外维度
                data_array = np.squeeze(data_array)
                # 确保是 [C, *S]
                if data_array.ndim == 1:
                    data_array = np.expand_dims(data_array, axis=0)  # [C]
                while len(data_array.shape) < self.num_dimensions + 1:
                    data_array = np.expand_dims(data_array, axis=-1)  # [C, 1, ...]

        # 应用变换
        for t in self.transforms:
            data_array = t(data_array)

        # 转换为张量
        data_tensor = torch.tensor(data_array, dtype=torch.float32)

        # 确保形状为 [C, *S]
        if data_tensor.dim() == self.num_dimensions:
            data_tensor = data_tensor.unsqueeze(0)  # [1, *S]
        elif data_tensor.dim() != self.num_dimensions + 1:
            raise ValueError(f"Unexpected data_tensor dim: {data_tensor.dim()}, expected {self.num_dimensions + 1}")

        # 检查与目标形状是否一致
        expected_shape = self.target_shape  # [C, *S]
        target_channels = self.target_shape[0]
        target_spatial = self.target_shape[1:]

        # 验证空间维度
        current_spatial = data_tensor.shape[1:]
        if current_spatial != target_spatial:
            raise ValueError(f"Spatial shape {current_spatial} does not match target {target_spatial}")

        # 处理通道维度
        current_channels = data_tensor.shape[0]
        if current_channels != target_channels:
            if target_channels > 1 and current_channels == 1:
                # 通道扩展（类似 one-hot 编码，适用于多分类/多分割）
                data_tensor = data_tensor.squeeze(0).long()  # [*S]
                data_tensor = F.one_hot(data_tensor, num_classes=target_channels)  # [*S, C]
                # 调整维度顺序为 [C, *S]
                permute_order = [-1] + list(range(self.num_dimensions))
                data_tensor = data_tensor.permute(*permute_order).float()
            elif current_channels > target_channels:
                # 裁剪通道
                data_tensor = data_tensor[:target_channels]
            else:
                raise ValueError(f"Cannot match channels: current {current_channels}, target {target_channels}")

        # 最终形状验证
        if data_tensor.shape != expected_shape:
            raise ValueError(f"Data tensor shape {data_tensor.shape} does not match target shape {expected_shape}")

        return data_tensor
