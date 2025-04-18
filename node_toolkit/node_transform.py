import numpy as np
from scipy.ndimage import rotate, zoom

class MinMaxNormalize:
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def __call__(self, img):
        # 输入 img: [C, *S]
        for i in range(img.shape[0]):
            channel = img[i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val != min_val:
                img[i] = (channel - min_val) / (max_val - min_val)
            else:
                img[i] = np.zeros_like(channel)
        return img

class ZScoreNormalize:
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def __call__(self, img):
        # 输入 img: [C, *S]
        for i in range(img.shape[0]):
            channel = img[i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            if std_val != 0:
                img[i] = (channel - mean_val) / std_val
            else:
                img[i] = np.zeros_like(channel)
        return img

class RandomRotate:
    def __init__(self, num_dimensions, max_angle=5):
        self.num_dimensions = num_dimensions
        self.max_angle = max_angle
        self.angles = None

    def __call__(self, img):
        # 输入 img: [C, *S]
        if self.angles is None:
            self.angles = np.random.uniform(-self.max_angle, self.max_angle, self.num_dimensions)
        for i, angle in enumerate(self.angles):
            axes = [(j % self.num_dimensions, (j + 1) % self.num_dimensions) for j in range(i, i + 2)][0]
            # 调整轴索引，跳过 channel 维度
            axes = (axes[0] + 1, axes[1] + 1)
            img = rotate(img, angle=angle, axes=axes, reshape=False, mode='nearest')
        return img

class RandomFlip:
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.flip_axes = None

    def __call__(self, img):
        # 输入 img: [C, *S]
        if self.flip_axes is None:
            self.flip_axes = [np.random.rand() < 0.5 for _ in range(self.num_dimensions)]
        for axis, flip in enumerate(self.flip_axes):
            if flip:
                # 从 axis=1 开始，跳过 channel 维度
                img = np.flip(img, axis=axis + 1).copy()
        return img

class RandomShift:
    def __init__(self, num_dimensions, max_shift=5):
        self.num_dimensions = num_dimensions
        self.max_shift = max_shift
        self.shifts = None

    def __call__(self, img):
        # 输入 img: [C, *S]
        if self.shifts is None:
            self.shifts = np.random.randint(-self.max_shift, self.max_shift, self.num_dimensions)
        for axis, shift in enumerate(self.shifts):
            # 从 axis=1 开始，跳过 channel 维度
            img = np.roll(img, shift, axis=axis + 1)
        return img

class RandomZoom:
    def __init__(self, num_dimensions, zoom_range=(0.9, 1.1)):
        self.num_dimensions = num_dimensions
        self.zoom_range = zoom_range
        self.zoom_factor = None

    def __call__(self, img):
        # 输入 img: [C, *S]
        if self.zoom_factor is None:
            self.zoom_factor = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoomed = np.zeros_like(img)
        for i in range(img.shape[0]):
            zoomed_slice = zoom(img[i], self.zoom_factor, mode='nearest')
            zoomed_slice = self._adjust_size(zoomed_slice, img.shape[1:])
            zoomed[i] = zoomed_slice
        return zoomed

    def _adjust_size(self, zoomed_slice, target_shape):
        # 输入 zoomed_slice: [*S], target_shape: [*S]
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

class RandomMask:
    def __init__(self, num_dimensions, mask_prob=0.2):
        self.num_dimensions = num_dimensions
        self.mask_prob = mask_prob
        self.mask = None

    def __call__(self, img):
        # 输入 img: [C, *S]
        if self.mask is None:
            self.mask = np.random.rand() < self.mask_prob
        if self.mask:
            return np.zeros_like(img)
        return img
