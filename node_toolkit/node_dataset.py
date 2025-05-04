"""
MHD_Nodet Project - Dataset Module
==================================
This module defines data loading and transformation utilities for the MHD_Nodet project,
including normalization, augmentation, and the NodeDataset class for loading medical imaging data.

项目：MHD_Nodet - 数据集模块
本模块定义了 MHD_Nodet 项目的数据加载和转换工具，包括归一化、数据增强和 NodeDataset 类，
用于加载医学影像数据。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import rotate, zoom
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinMaxNormalize:
    """
    Min-Max normalization for image data.
    图像数据的 Min-Max 归一化。
    """
    def __call__(self, img):
        input_dtype = img.dtype
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            channel = img[i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val != min_val:
                img[i] = (channel - min_val) / (max_val - min_val)
            else:
                img[i] = np.zeros_like(channel)
        if np.issubdtype(input_dtype, np.integer):
            img = np.clip(img * 255, 0, 255).astype(input_dtype)
        else:
            img = img.astype(input_dtype)
        return img

    def reset(self):
        pass

class ZScoreNormalize:
    """
    Z-Score normalization for image data.
    图像数据的 Z-Score 归一化。
    """
    def __call__(self, img):
        input_dtype = img.dtype
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            channel = img[i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            if std_val != 0:
                img[i] = (channel - mean_val) / std_val
            else:
                img[i] = np.zeros_like(channel)
        if np.issubdtype(input_dtype, np.integer):
            img = np.clip(img * 255, -255, 255).astype(input_dtype)
        else:
            img = img.astype(input_dtype)
        return img

    def reset(self):
        pass

class RandomRotate:
    """
    Random rotation augmentation with batch-consistent randomness.
    带批次一致随机性的随机旋转增强。
    """
    def __init__(self, max_angle=5):
        self.max_angle = max_angle
        self.angles = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        """
        Set the seed for batch-consistent random parameters, ensuring consistency across workers.
        设置批次一致随机参数的种子，确保跨worker一致。
        """
        self.batch_seed = seed
        self.angles = None
        np.random.seed(seed)
        num_dims = 3  # Assuming 3D data
        self.angles = np.random.uniform(-self.max_angle, self.max_angle, num_dims)
        logger.debug(f"RandomRotate: Seed {seed}, Angles {self.angles}")

    def __call__(self, img):
        input_dtype = img.dtype
        is_integer = np.issubdtype(input_dtype, np.integer)
        order = 0 if is_integer else 1
        num_dims = len(img.shape) - 1
        if self.angles is None:
            raise ValueError("Batch seed not set for RandomRotate")
        for i, angle in enumerate(self.angles):
            axes = [(j % num_dims, (j + 1) % num_dims) for j in range(i, i + 2)][0]
            axes = (axes[0] + 1, axes[1] + 1)
            img = rotate(img, angle=angle, axes=axes, reshape=False, order=order, mode='nearest')
        img = img.astype(input_dtype)
        return img

    def reset(self):
        self.angles = None

class RandomFlip:
    """
    Random flip augmentation with batch-consistent randomness.
    带批次一致随机性的随机翻转增强。
    """
    def __init__(self):
        self.flip_axes = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        """
        Set the seed for batch-consistent random parameters, ensuring consistency across workers.
        设置批次一致随机参数的种子，确保跨worker一致。
        """
        self.batch_seed = seed
        self.flip_axes = None
        np.random.seed(seed)
        num_dims = 3
        self.flip_axes = [np.random.rand() < 0.5 for _ in range(num_dims)]
        logger.debug(f"RandomFlip: Seed {seed}, Flip axes {self.flip_axes}")

    def __call__(self, img):
        input_dtype = img.dtype
        num_dims = len(img.shape) - 1
        if self.flip_axes is None:
            raise ValueError("Batch seed not set for RandomFlip")
        for axis, flip in enumerate(self.flip_axes):
            if flip:
                img = np.flip(img, axis=axis + 1).copy()
        img = img.astype(input_dtype)
        return img

    def reset(self):
        self.flip_axes = None

class RandomShift:
    """
    Random shift augmentation with batch-consistent randomness.
    带批次一致随机性的随机平移增强。
    """
    def __init__(self, max_shift=5):
        self.max_shift = max_shift
        self.shifts = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        """
        Set the seed for batch-consistent random parameters, ensuring consistency across workers.
        设置批次一致随机参数的种子，确保跨worker一致。
        """
        self.batch_seed = seed
        self.shifts = None
        np.random.seed(seed)
        num_dims = 3
        self.shifts = np.random.randint(-self.max_shift, self.max_shift, num_dims)
        logger.debug(f"RandomShift: Seed {seed}, Shifts {self.shifts}")

    def __call__(self, img):
        input_dtype = img.dtype
        num_dims = len(img.shape) - 1
        if self.shifts is None:
            raise ValueError("Batch seed not set for RandomShift")
        for axis, shift in enumerate(self.shifts):
            img = np.roll(img, shift, axis=axis + 1)
        img = img.astype(input_dtype)
        return img

    def reset(self):
        self.shifts = None

class RandomZoom:
    """
    Random zoom augmentation with batch-consistent randomness.
    带批次一致随机性的随机缩放增强。
    """
    def __init__(self, zoom_range=(0.9, 1.1)):
        self.zoom_range = zoom_range
        self.zoom_factor = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        """
        Set the seed for batch-consistent random parameters, ensuring consistency across workers.
        设置批次一致随机参数的种子，确保跨worker一致。
        """
        self.batch_seed = seed
        self.zoom_factor = None
        np.random.seed(seed)
        self.zoom_factor = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        logger.debug(f"RandomZoom: Seed {seed}, Zoom factor {self.zoom_factor}")

    def __call__(self, img):
        input_dtype = img.dtype
        is_integer = np.issubdtype(input_dtype, np.integer)
        order = 0 if is_integer else 1
        if self.zoom_factor is None:
            raise ValueError("Batch seed not set for RandomZoom")
        zoomed = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[0]):
            zoomed_slice = zoom(img[i], self.zoom_factor, order=order, mode='nearest')
            zoomed_slice = self._adjust_size(zoomed_slice, img.shape[1:])
            zoomed[i] = zoomed_slice
        if is_integer:
            zoomed = np.clip(zoomed, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        else:
            zoomed = zoomed.astype(input_dtype)
        return zoomed

    def _adjust_size(self, zoomed_slice, target_shape):
        for dim in range(len(target_shape)):
            if zoomed_slice.shape[dim] != target_shape[dim]:
                if zoomed_slice.shape[dim] > target_shape[dim]:
                    start = (zoomed_slice.shape[dim] - target_shape[dim]) // 2
                    end = start + target_shape[dim]
                    zoomed_slice = np.take(zoomed_slice, np.arange(start, end), axis=dim)
                else:
                    pad_width = [(0, 0)] * len(target_shape)
                    pad_width[dim] = ((target_shape[dim] - zoomed_slice.shape[dim]) + 1) // 2, (target_shape[dim] - zoomed_slice.shape[dim]) // 2
                    zoomed_slice = np.pad(zoomed_slice, pad_width, mode='constant', constant_values=0)
        return zoomed_slice

    def reset(self):
        self.zoom_factor = None

class NodeDataset(Dataset):
    """
    Dataset class for loading medical imaging data with transformations.
    用于加载医学影像数据并应用变换的数据集类。
    """
    def __init__(self, data_dir, node_id, suffix, target_shape, transforms=None, node_mapping=None, sub_networks=None, case_ids=None, case_id_order=None, batch_seed=None):
        self.data_dir = data_dir
        self.node_id = node_id
        self.suffix = suffix
        self.target_shape = target_shape
        self.transforms = transforms or []
        self.node_mapping = node_mapping
        self.sub_networks = sub_networks
        self.case_id_order = case_id_order
        self.batch_seed = batch_seed

        self.dtype = self._get_node_dtype()
        self.np_dtype = np.int64 if self.dtype == torch.int64 else np.float32

        all_files = sorted(os.listdir(data_dir))
        file_ext = None
        for case_id in (case_ids or ['0000']):
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

        self.case_ids = []
        for case_id in case_ids or []:
            if f'case_{case_id}_{self.suffix}{self.file_ext}' in all_files:
                self.case_ids.append(case_id)
        if not self.case_ids:
            raise ValueError(f"No valid case IDs found for suffix {self.suffix} in {data_dir}")

        if self.case_id_order is not None:
            invalid_ids = [cid for cid in self.case_id_order if cid not in self.case_ids]
            if invalid_ids:
                raise ValueError(f"Invalid case IDs in case_id_order: {invalid_ids}")
            self.case_ids = self.case_id_order
        else:
            self.case_ids = sorted(self.case_ids)

    def _get_node_dtype(self):
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            if global_node == self.node_id and sub_net_name in self.sub_networks:
                sub_net = self.sub_networks[sub_net_name]
                dtype_str = sub_net.node_dtype.get(sub_node_id, "float")
                return torch.int64 if dtype_str == "long" else torch.float32
        raise ValueError(f"Node {self.node_id} not found in node_mapping or invalid sub_network")

    def set_batch_seed(self, seed):
        """
        Set the seed for batch-consistent random transformations, ensuring consistency across workers.
        设置批次一致随机变换的种子，确保跨worker一致。
        """
        self.batch_seed = seed
        for t in self.transforms:
            if hasattr(t, 'set_batch_seed'):
                t.set_batch_seed(seed)
        logger.debug(f"NodeDataset {self.node_id}: Set batch seed {seed}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        data_path = os.path.join(self.data_dir, f'case_{case_id}_{self.suffix}{self.file_ext}')

        if self.file_ext == '.nii.gz':
            data = nib.load(data_path).get_fdata()
            data_array = np.asarray(data, dtype=np.float32)
            data_array = np.squeeze(data_array)
            if len(data_array.shape) < len(self.target_shape) - 1:
                data_array = np.expand_dims(data_array, axis=0)
            if len(data_array.shape) == len(self.target_shape) - 1:
                data_array = np.expand_dims(data_array, axis=0)
        else:
            df = pd.read_csv(data_path)
            if 'Value' not in df.columns:
                raise ValueError(f"CSV file {data_path} does not have 'Value' column")
            value = df['Value'].iloc[0]
            if isinstance(value, (int, float)):
                data_array = np.full([1] + list(self.target_shape[1:]), float(value), dtype=np.float32)
            else:
                data_array = np.array(value, dtype=np.float32)
                data_array = np.squeeze(data_array)
                if data_array.ndim == 1:
                    data_array = np.expand_dims(data_array, axis=0)
                while len(data_array.shape) < len(self.target_shape):
                    data_array = np.expand_dims(data_array, axis=-1)

        if self.dtype == torch.int64:
            data_array = np.round(data_array).astype(np.int64)
            data_array = np.clip(data_array, 0, self.target_shape[0] - 1)
        else:
            data_array = data_array.astype(self.np_dtype)

        for t in self.transforms:
            data_array = t(data_array)

        data_tensor = torch.tensor(data_array, dtype=self.dtype)

        if data_tensor.dim() == len(self.target_shape) - 1:
            data_tensor = data_tensor.unsqueeze(0)
        elif data_tensor.dim() != len(self.target_shape):
            raise ValueError(f"Unexpected data_tensor dim: {data_tensor.dim()}, expected {len(self.target_shape)}")

        current_spatial = data_tensor.shape[1:]
        target_spatial = self.target_shape[1:]
        if current_spatial != target_spatial:
            raise ValueError(f"Spatial shape {current_spatial} does not match target {target_spatial} for node {self.node_id}")

        current_channels = data_tensor.shape[0]
        target_channels = self.target_shape[0]
        if current_channels != target_channels:
            if self.dtype == torch.int64 and target_channels > 1 and current_channels == 1:
                data_tensor = data_tensor.squeeze(0).long()
                data_tensor = F.one_hot(data_tensor, num_classes=target_channels)
                permute_order = [-1] + list(range(len(self.target_shape) - 1))
                data_tensor = data_tensor.permute(*permute_order).float()
            else:
                raise ValueError(f"Cannot match channels: current {current_channels}, target {target_channels} for node {self.node_id}")

        if data_tensor.shape != self.target_shape:
            raise ValueError(f"Data tensor shape {data_tensor.shape} does not match target shape {self.target_shape} for node {self.node_id}")

        return data_tensor

