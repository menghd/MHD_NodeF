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
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import rotate, zoom
import logging
from typing import List, Tuple, Iterable, Set, Dict, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderedSampler(Sampler):
    """保持全局顺序的采样器（所有 worker 都返回同一顺序）"""
    def __init__(self, indices, num_workers):
        self.indices = indices
        self.num_workers = max(1, num_workers)
        logger.info(f"OrderedSampler: Total indices {len(self.indices)}, num_workers {self.num_workers}")

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def worker_init_fn(worker_id):
    """为每个 worker 设置唯一种子，保证可复现"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2 ** 32)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        logger.info(f"Worker {worker_id} initialized with seed {seed}")


# ------------------- 归一化 / 增强 -------------------
class MinMaxNormalize:
    def __call__(self, img):
        input_dtype = img.dtype
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            ch = img[i]
            mn, mx = np.min(ch), np.max(ch)
            img[i] = (ch - mn) / (mx - mn) if mx != mn else np.zeros_like(ch)
        if np.issubdtype(input_dtype, np.integer):
            img = np.clip(img * 255, 0, 255).astype(input_dtype)
        else:
            img = img.astype(input_dtype)
        return img

    def reset(self): pass


class ZScoreNormalize:
    def __call__(self, img):
        input_dtype = img.dtype
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            ch = img[i]
            mean, std = np.mean(ch), np.std(ch)
            img[i] = (ch - mean) / std if std != 0 else np.zeros_like(ch)
        if np.issubdtype(input_dtype, np.integer):
            img = np.clip(img * 255, -255, 255).astype(input_dtype)
        else:
            img = img.astype(input_dtype)
        return img

    def reset(self): pass


class OneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.batch_seed = None

    def set_batch_seed(self, seed):
        self.batch_seed = seed

    def __call__(self, img):
        img = np.round(img).astype(np.int64)
        img = np.clip(img, 0, self.num_classes - 1)
        t = torch.tensor(img, dtype=torch.int64)
        if t.ndim > 1:
            t = t.squeeze()
        t = F.one_hot(t, num_classes=self.num_classes)
        perm = [-1] + list(range(t.ndim - 1))
        return t.permute(*perm).float().numpy().astype(np.float32)

    def reset(self): pass


class RandomFlip:
    def __init__(self, p=0.5, axes=None):
        self.p = p
        self.axes = axes
        self.flip_axis = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        self.batch_seed = seed
        self.flip_axis = None

    def __call__(self, img):
        input_dtype = img.dtype
        ndim = len(img.shape) - 1
        if self.flip_axis is None:
            np.random.seed(self.batch_seed)
            axes = self.axes if self.axes is not None else list(range(ndim))
            self.flip_axis = np.random.choice(axes) if np.random.rand() < self.p else None
        if self.flip_axis is not None:
            img = np.flip(img, axis=self.flip_axis + 1).copy()
        return img.astype(input_dtype)

    def reset(self):
        self.flip_axis = None


class RandomRotate:
    def __init__(self, max_angle=5, p=0.5):
        self.max_angle = max_angle
        self.p = p
        self.angles = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        self.batch_seed = seed
        self.angles = None

    def __call__(self, img):
        input_dtype = img.dtype
        is_int = np.issubdtype(input_dtype, np.integer)
        order = 0 if is_int else 1
        ndim = len(img.shape) - 1
        np.random.seed(self.batch_seed)
        if np.random.rand() >= self.p:
            return img.astype(input_dtype)

        if self.angles is None:
            if ndim == 2:
                self.angles = [np.random.uniform(-self.max_angle, self.max_angle)]
            elif ndim == 3:
                self.angles = np.random.uniform(-self.max_angle, self.max_angle, 3)
            else:
                raise ValueError(f"Unsupported ndim {ndim}")

        if ndim == 2:
            img = rotate(img, self.angles[0], axes=(1, 2), reshape=False, order=order, mode='nearest')
        else:
            for i, ang in enumerate(self.angles):
                ax = ((i % ndim) + 1, ((i + 1) % ndim) + 1)
                img = rotate(img, ang, axes=ax, reshape=False, order=order, mode='nearest')
        return img.astype(input_dtype)

    def reset(self):
        self.angles = None


class RandomShift:
    def __init__(self, max_shift=5, p=0.5):
        self.max_shift = max_shift
        self.p = p
        self.shifts = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        self.batch_seed = seed
        self.shifts = None

    def __call__(self, img):
        input_dtype = img.dtype
        ndim = len(img.shape) - 1
        np.random.seed(self.batch_seed)
        if np.random.rand() >= self.p:
            return img.astype(input_dtype)

        if self.shifts is None:
            self.shifts = np.random.randint(-self.max_shift, self.max_shift + 1, ndim)
        for ax, sh in enumerate(self.shifts):
            img = np.roll(img, sh, axis=ax + 1)
        return img.astype(input_dtype)

    def reset(self):
        self.shifts = None


class RandomZoom:
    def __init__(self, zoom_range=(0.9, 1.1), p=0.5):
        self.zoom_range = zoom_range
        self.p = p
        self.zoom_factor = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        self.batch_seed = seed
        self.zoom_factor = None

    def __call__(self, img):
        input_dtype = img.dtype
        is_int = np.issubdtype(input_dtype, np.integer)
        order = 0 if is_int else 1
        ndim = len(img.shape) - 1
        np.random.seed(self.batch_seed)
        if np.random.rand() >= self.p:
            return img.astype(input_dtype)

        if self.zoom_factor is None:
            self.zoom_factor = np.random.uniform(*self.zoom_range)

        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[0]):
            z = zoom(img[c], self.zoom_factor, order=order, mode='nearest')
            out[c] = self._adjust(z, img.shape[1:])
        if is_int:
            out = np.clip(out, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        else:
            out = out.astype(input_dtype)
        return out

    def _adjust(self, z, target):
        for d in range(len(target)):
            if z.shape[d] != target[d]:
                if z.shape[d] > target[d]:
                    s = (z.shape[d] - target[d]) // 2
                    z = np.take(z, np.arange(s, s + target[d]), axis=d)
                else:
                    pad = [(0, 0)] * len(target)
                    diff = target[d] - z.shape[d]
                    pad[d] = (diff + 1) // 2, diff // 2
                    z = np.pad(z, pad, mode='constant')
        return z

    def reset(self):
        self.zoom_factor = None


class RandomDrop:
    def __init__(self, p=0.5):
        self.p = p
        self.drop = None
        self.batch_seed = None

    def set_batch_seed(self, seed):
        self.batch_seed = seed
        self.drop = None

    def __call__(self, img):
        input_dtype = img.dtype
        if self.drop is None:
            np.random.seed(self.batch_seed)
            self.drop = np.random.rand() < self.p
        return np.zeros_like(img, dtype=input_dtype) if self.drop else img

    def reset(self):
        self.drop = None


# ------------------- Dataset -------------------
class NodeDataset(Dataset):
    """
    加载 NIfTI / CSV，支持 set/list 形式的 load_node。
    """
    def __init__(
        self,
        data_dir: str,
        node_id: str,
        filename: str,
        target_shape: Tuple[int, ...],
        transforms: Optional[List] = None,
        node_mapping: Optional[List] = None,
        sub_networks: Optional[Dict] = None,
        case_ids: Optional[List[str]] = None,
        case_id_order: Optional[List[str]] = None,
        num_dimensions: int = 3,
        batch_seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.node_id = str(node_id)
        self.filename = filename
        self.target_shape = target_shape
        self.transforms = transforms or []
        self.node_mapping = node_mapping
        self.sub_networks = sub_networks
        self.case_id_order = case_id_order
        self.num_dimensions = num_dimensions
        self.batch_seed = batch_seed

        all_files = sorted(os.listdir(data_dir))
        self.file_ext = '.' + filename.split('.', 1)[1] if '.' in filename else ''

        # ---------- 接受 case_ids ----------
        self.case_ids = sorted(case_ids or [])
        if not self.case_ids:
            logger.warning(f"No case IDs provided for node {self.node_id}, filename {self.filename}")
            self.case_ids = []

        # ---------- 统一顺序 ----------
        if self.case_id_order is not None:
            invalid = [c for c in self.case_id_order if c not in self.case_ids]
            if invalid:
                raise ValueError(f"Invalid case IDs in case_id_order: {invalid}")
            self.case_ids = self.case_id_order
        else:
            self.case_ids = sorted(self.case_ids)

        # ---------- 检查缺失文件 ----------
        missing = [c for c in self.case_ids
                   if f'case_{c}_{self.filename}' not in all_files]
        if missing:
            logger.warning(f"Missing files for node {self.node_id}, filename {self.filename}: {missing}")

    def set_batch_seed(self, seed):
        self.batch_seed = seed
        for t in self.transforms:
            if hasattr(t, 'set_batch_seed'):
                t.set_batch_seed(seed)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        path = os.path.join(self.data_dir, f'case_{case_id}_{self.filename}')

        # ---------- 读取 ----------
        if not os.path.exists(path):
            logger.warning(f"File missing: {path}. Generating placeholder.")
            data_array = np.zeros([1] + list(self.target_shape[1:]), dtype=np.float32)
        else:
            if self.file_ext == '.nii.gz':
                data = nib.load(path).get_fdata()
                data_array = np.asarray(data, dtype=np.float32).squeeze()
                if data_array.ndim == self.num_dimensions:
                    data_array = np.expand_dims(data_array, 0)
                elif data_arrayylen(data_array.ndim) < self.num_dimensions:
                    raise ValueError(f"NIfTI ndim {data_array.ndim} < {self.num_dimensions}")
            elif self.file_ext == '.csv':
                df = pd.read_csv(path)
                if 'Value' not in df.columns:
                    raise ValueError(f"CSV {path} missing 'Value' column")
                val = df['Value'].iloc[0]
                data_array = np.full([1] + list(self.target_shape[1:]), float(val), dtype=np.float32)
            else:
                raise ValueError(f"Unsupported extension {self.file_ext}")

        # ---------- 变换 ----------
        for t in self.transforms:
            data_array = t(data_array)

        tensor = torch.tensor(data_array, dtype=torch.float32)

        # ---------- 通道/空间对齐 ----------
        if tensor.shape[0] != self.target_shape[0]:
            if isinstance(self.transforms[-1], OneHot):
                if tensor.shape[0] != self.target_shape[0]:
                    raise ValueError(f"OneHot produced {tensor.shape[0]} channels, want {self.target_shape[0]}")
            else:
                raise ValueError(f"Channel mismatch for node {self.node_id}")

        # 补齐/压缩维度
        cur_spatial = tensor.shape[1:]
        exp_spatial = len(self.target_shape) - 1
        if len(cur_spatial) != exp_spatial:
            if len(cur_spatial) < exp_spatial:
                for _ in range(exp_spatial - len(cur_spatial)):
                    tensor = tensor.unsqueeze(-1)
            else:
                for d in range(len(cur_spatial) - 1, exp_spatial - 1, -1):
                    if tensor.shape[d] == 1:
                        tensor = tensor.squeeze(d)
                    else:
                        raise ValueError(f"Cannot squeeze dim {d} size {tensor.shape[d]}")

        # padding to exact target size
        cur_spatial = tensor.shape[1:]
        pad = []
        for c, t in zip(cur_spatial, self.target_shape[1:]):
            if c < t:
                p1 = (t - c) // 2
                p2 = t - c - p1
                pad.extend([p1, p2])
            elif c > t:
                raise ValueError(f"Spatial dim {c} > target {t} for node {self.node_id}")
            else:
                pad.extend([0, 0])
        if any(p != 0 for p in pad):
            tensor = F.pad(tensor, pad[::-1], mode='constant', value=0)

        if tensor.shape != self.target_shape:
            raise ValueError(f"Final shape {tensor.shape} != target {self.target_shape}")

        return tensor


# ------------------- 工具函数 -------------------
def create_dataloaders(
    data_dir: str,
    load_node: Union[Iterable[Tuple[str, str]], Set[Tuple[str, str]]],
    in_nodes: Union[List[str], Set[str]],
    target_shapes: Dict[str, Tuple[int, ...]],
    transforms_dict: Optional[Dict[str, List]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    case_ids: Optional[List[str]] = None,
    case_id_order: Optional[List[str]] = None,
    num_dimensions: int = 3,
) -> Dict[str, DataLoader]:
    """
    统一创建所有节点的 DataLoader。

    Parameters
    ----------
    load_node : set / list / iterable
        (node_id, filename) 集合，**支持 set**。
    in_nodes : set / list
        需要加载的节点集合（用于过滤）。
    target_shapes : dict
        {node_id: target_shape}。
    transforms_dict : dict, optional
        {node_id: [transform, ...]}。
    """
    load_node = list(load_node)                     # 统一转 list
    in_nodes = list(in_nodes)

    dataloaders = {}
    for node, filename in load_node:
        if node not in in_nodes:
            continue
        shape = target_shapes.get(node)
        if shape is None:
            raise ValueError(f"Missing target_shape for node {node}")
        tfms = transforms_dict.get(node, []) if transforms_dict else []
        ds = NodeDataset(
            data_dir=data_dir,
            node_id=node,
            filename=filename,
            target_shape=shape,
            transforms=tfms,
            case_ids=case_ids,
            case_id_order=case_id_order,
            num_dimensions=num_dimensions,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=OrderedSampler(list(range(len(ds))), num_workers),
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )
        dataloaders[node] = dl
    return dataloaders
