"""
MHD_Nodet Project - Dataset Module
==================================
This module provides data loading and preprocessing utilities for the MHD_Nodet project, designed for medical imaging data.
- Includes dataset class (NodeDataset) for loading NIfTI and CSV files with customizable transformations.
- Supports batch-consistent data augmentations (rotation, flip, shift, zoom) and normalization (Min-Max, Z-Score).
- Includes OrderedSampler for consistent data ordering and worker_init_fn for reproducible worker initialization.

项目：MHD_Nodet - 数据集模块
本模块为 MHD_Nodet 项目提供数据加载和预处理工具，专为医学影像数据设计。
- 包含 NodeDataset 类，用于加载 NIfTI 和 CSV 文件，支持自定义变换。
- 支持批次一致的数据增强（旋转、翻转、平移、缩放）和归一化（Min-Max、Z-Score）。
- 包含 OrderedSampler 类以确保数据顺序一致，以及 worker_init_fn 以确保可重现的worker初始化。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import rotate, zoom
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderedSampler(Sampler):
    """
    Custom sampler to enforce consistent order of indices across workers.
    Ensures all workers process the entire dataset in the same order.
    自定义采样器以强制执行一致的索引顺序，确保所有worker处理整个数据集。
    """
    def __init__(self, indices, num_workers):
        self.indices = indices
        self.num_workers = max(1, num_workers)
        logger.info(f"OrderedSampler: Total indices {len(self.indices)}, num_workers {self.num_workers}")

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def worker_init_fn(worker_id):
    """
    Initialize worker with a unique seed for reproducibility, ensuring consistent random state across workers.
    使用唯一种子初始化工作进程以确保可重现性。
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2**32)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        logger.info(f"Worker {worker_id} initialized with seed {seed}")

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

class OneHot:
    """
    One-hot encoding transformation for integer or float data.
    针对整型或浮点型数据的独热编码变换。
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.batch_seed = None

    def set_batch_seed(self, seed):
        """
        Set the seed for batch-consistent random parameters.
        设置批次一致随机参数的种子。
        """
        self.batch_seed = seed
        logger.debug(f"OneHot: Set batch seed {seed}")

    def __call__(self, img):
        img = img.astype(np.float32)
        # Convert to integer by rounding
        img = np.round(img).astype(np.int64)
        # Clip to valid class range
        img = np.clip(img, 0, self.num_classes - 1)
        # Convert to one-hot
        img_tensor = torch.tensor(img, dtype=torch.int64)
        if img_tensor.ndim > 1:
            img_tensor = img_tensor.squeeze()
        img_tensor = F.one_hot(img_tensor, num_classes=self.num_classes)
        # Permute to [C, ...] format
        permute_order = [-1] + list(range(img_tensor.ndim - 1))
        img_tensor = img_tensor.permute(*permute_order).float()
        # Ensure output is float32
        return img_tensor.numpy().astype(np.float32)

    def reset(self):
        pass

class RandomRotate:
    """
    Random rotation augmentation with batch-consistent randomness, supporting 2D or 3D data.
    带批次一致随机性的随机旋转增强，支持 2D 或 3D 数据。
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
        logger.debug(f"RandomRotate: Set batch seed {seed}")

    def __call__(self, img):
        input_dtype = img.dtype
        is_integer = np.issubdtype(input_dtype, np.integer)
        order = 0 if is_integer else 1
        num_dims = len(img.shape) - 1  # Spatial dimensions

        if self.angles is None:
            np.random.seed(self.batch_seed)
            if num_dims == 2:
                # For 2D, only rotate in the single plane
                self.angles = [np.random.uniform(-self.max_angle, self.max_angle)]
            elif num_dims == 3:
                # For 3D, rotate in three planes
                self.angles = np.random.uniform(-self.max_angle, self.max_angle, 3)
            else:
                raise ValueError(f"Unsupported number of dimensions: {num_dims}")
            logger.debug(f"RandomRotate: Angles {self.angles}")

        if num_dims == 2:
            # 2D rotation: single angle
            img = rotate(img, angle=self.angles[0], axes=(1, 2), reshape=False, order=order, mode='nearest')
        elif num_dims == 3:
            # 3D rotation: apply rotations sequentially
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
    Random flip augmentation with batch-consistent randomness, supporting 2D or 3D data.
    带批次一致随机性的随机翻转增强，支持 2D 或 3D 数据。
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
        self.flip_axes = None
        logger.debug(f"RandomFlip: Set batch seed {seed}")

    def __call__(self, img):
        input_dtype = img.dtype
        num_dims = len(img.shape) - 1  # Spatial dimensions

        if self.flip_axes is None:
            np.random.seed(self.batch_seed)
            self.flip_axes = [np.random.rand() < 0.5 for _ in range(num_dims)]
            logger.debug(f"RandomFlip: Flip axes {self.flip_axes}")

        for axis, flip in enumerate(self.flip_axes):
            if flip:
                img = np.flip(img, axis=axis + 1).copy()
        img = img.astype(input_dtype)
        return img

    def reset(self):
        self.flip_axes = None

class RandomShift:
    """
    Random shift augmentation with batch-consistent randomness, supporting 2D or 3D data.
    带批次一致随机性的随机平移增强，支持 2D 或 3D 数据。
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
        logger.debug(f"RandomShift: Set batch seed {seed}")

    def __call__(self, img):
        input_dtype = img.dtype
        num_dims = len(img.shape) - 1  # Spatial dimensions

        if self.shifts is None:
            np.random.seed(self.batch_seed)
            self.shifts = np.random.randint(-self.max_shift, self.max_shift, num_dims)
            logger.debug(f"RandomShift: Shifts {self.shifts}")

        for axis, shift in enumerate(self.shifts):
            img = np.roll(img, shift, axis=axis + 1)
        img = img.astype(input_dtype)
        return img

    def reset(self):
        self.shifts = None

class RandomZoom:
    """
    Random zoom augmentation with batch-consistent randomness, supporting 2D or 3D data.
    带批次一致随机性的随机缩放增强，支持 2D 或 3D 数据。
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
        num_dims = len(img.shape) - 1  # Spatial dimensions

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
    def __init__(self, data_dir, node_id, suffix, target_shape, transforms=None, node_mapping=None, sub_networks=None, case_ids=None, case_id_order=None, num_dimensions=3, batch_seed=None):
        self.data_dir = data_dir
        self.node_id = node_id
        self.suffix = suffix
        self.target_shape = target_shape
        self.transforms = transforms or []
        self.node_mapping = node_mapping
        self.sub_networks = sub_networks
        self.case_id_order = case_id_order
        self.num_dimensions = num_dimensions
        self.batch_seed = batch_seed

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
            # Load 3D NIfTI data
            data = nib.load(data_path).get_fdata()
            data_array = np.asarray(data, dtype=np.float32)
            data_array = np.squeeze(data_array)
            # Ensure at least one channel dimension
            if data_array.ndim == self.num_dimensions:
                data_array = np.expand_dims(data_array, axis=0)
            elif data_array.ndim < self.num_dimensions:
                raise ValueError(f"NIfTI data has fewer dimensions ({data_array.ndim}) than expected ({self.num_dimensions})")
        else:
            # Load CSV data (1D)
            df = pd.read_csv(data_path)
            if 'Value' not in df.columns:
                raise ValueError(f"CSV file {data_path} does not have 'Value' column")
            value = df['Value'].iloc[0]
            # Create a 1D array with the value, matching target spatial dimensions
            data_array = np.full([1] + list(self.target_shape[1:]), float(value), dtype=np.float32)

        # Apply transformations
        for t in self.transforms:
            data_array = t(data_array)

        # Convert to tensor
        data_tensor = torch.tensor(data_array, dtype=torch.float32)

        # Adjust dimensions to match target_shape
        current_shape = data_tensor.shape
        target_channels = self.target_shape[0]
        target_spatial = self.target_shape[1:]

        # Handle channel dimension
        if current_shape[0] != target_channels:
            if isinstance(self.transforms[-1], OneHot):
                # OneHot should have already set the correct channel dimension
                if current_shape[0] != target_channels:
                    raise ValueError(f"OneHot transform produced {current_shape[0]} channels, expected {target_channels}")
            else:
                raise ValueError(f"Channel mismatch: current {current_shape[0]}, target {target_channels} for node {self.node_id}")

        # Handle spatial dimensions
        current_spatial = current_shape[1:]
        expected_spatial_dims = len(self.target_shape) - 1
        if len(current_spatial) != expected_spatial_dims:
            # Adjust spatial dimensions by adding or removing singleton dimensions
            if len(current_spatial) < expected_spatial_dims:
                # Add singleton dimensions
                for _ in range(expected_spatial_dims - len(current_spatial)):
                    data_tensor = data_tensor.unsqueeze(-1)
            elif len(current_spatial) > expected_spatial_dims:
                # Remove extra singleton dimensions
                for dim in range(len(current_spatial) - 1, expected_spatial_dims - 1, -1):
                    if data_tensor.shape[dim] == 1:
                        data_tensor = data_tensor.squeeze(dim)
                    else:
                        raise ValueError(f"Non-singleton dimension {dim} with size {data_tensor.shape[dim]} cannot be squeezed")

        # Verify spatial dimensions match
        current_spatial = data_tensor.shape[1:]
        if current_spatial != target_spatial:
            # Pad or crop spatial dimensions to match target
            pad_width = []
            for curr, targ in zip(current_spatial, target_spatial):
                if curr < targ:
                    pad_before = (targ - curr) // 2
                    pad_after = targ - curr - pad_before
                    pad_width.append((pad_before, pad_after))
                elif curr > targ:
                    raise ValueError(f"Spatial dimension {curr} exceeds target {targ} for node {self.node_id}")
                else:
                    pad_width.append((0, 0))
            if any(p != (0, 0) for p in pad_width):
                data_tensor = F.pad(data_tensor, [p for pair in pad_width for p in pair], mode='constant', value=0)

        # Final shape validation
        if data_tensor.shape != self.target_shape:
            raise ValueError(f"Data tensor shape {data_tensor.shape} does not match target shape {self.target_shape} for node {self.node_id}")

        return data_tensor
