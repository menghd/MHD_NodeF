# MHD_Nodet Project

## 项目概述 / Project Overview

MHD_Nodet（Multi Hypergraph Dynamic Node Network）是一个基于多超图动态节点网络的深度学习框架，专为处理复杂的医学影像分割任务而设计。该项目通过模块化的网络架构（DNet、HDNet 和 MHDNet）实现动态超图处理，支持多任务学习（如分割和回归），并结合数据增强、归一化和交叉验证来提升模型性能。项目主要应用于医学影像分析，例如斑块分割等任务。

The MHD_Nodet (Multi Hypergraph Dynamic Node Network) project is a deep learning framework based on multi-hypergraph dynamic node networks, designed specifically for complex medical image segmentation tasks. It implements dynamic hypergraph processing through a modular network architecture (DNet, HDNet, and MHDNet), supports multi-task learning (e.g., segmentation and regression), and incorporates data augmentation, normalization, and cross-validation to enhance model performance. The project is primarily applied to medical image analysis, such as plaque segmentation.

---

## 项目特点 / Key Features

- **模块化网络架构 / Modular Network Architecture**：包括 DNet（动态卷积网络）、HDNet（超边网络）和 MHDNet（多子超图网络），支持灵活的节点连接和动态维度处理。
- **多任务支持 / Multi-Task Support**：通过任务配置支持分割、回归和分类任务，结合 Dice、IoU 和 Focal Loss 等损失函数。
- **数据增强与归一化 / Data Augmentation and Normalization**：提供随机旋转、翻转、平移、缩放以及 Min-Max 和 Z-Score 归一化，提升数据多样性和模型鲁棒性。
- **K 折交叉验证 / K-Fold Cross-Validation**：实现稳定的模型评估和性能优化。
- **医学影像优化 / Medical Imaging Optimization**：针对 NIfTI 和 CSV 格式的医学影像数据，提供高效的数据加载和预处理。

- **Modular Network Architecture**: Includes DNet (dynamic convolutional network), HDNet (hyperedge network), and MHDNet (multi-sub-hypergraph network), supporting flexible node connections and dynamic dimension handling.
- **Multi-Task Support**: Supports segmentation, regression, and classification tasks through task configurations, combined with loss functions like Dice, IoU, and Focal Loss.
- **Data Augmentation and Normalization**: Provides random rotation, flipping, shifting, zooming, and Min-Max/Z-Score normalization to enhance data diversity and model robustness.
- **K-Fold Cross-Validation**: Enables stable model evaluation and performance optimization.
- **Medical Imaging Optimization**: Optimized for NIfTI and CSV medical imaging data, offering efficient data loading and preprocessing.

---

## 文件结构 / Project Structure

项目代码组织如下，核心功能分布在 `node_pipline` 和 `node_toolkit` 目录中：

The project code is organized as follows, with core functionalities distributed in the `node_pipline` and `node_toolkit` directories:

```
C:\Users\souray\Desktop\Codes
|-- node_pipline
|   |-- node_train.py        # 训练流水线，包含数据准备、模型训练和交叉验证
|-- node_toolkit
|   |-- node_dataset.py      # 数据集模块，定义数据加载和增强工具
|   |-- node_net.py          # 网络模块，定义 DNet、HDNet 和 MHDNet 架构
|   |-- node_results.py      # 结果模块，定义损失函数和评估指标
|   |-- node_utils.py        # 工具模块，提供训练和验证的辅助函数
```

- **node_train.py**: Implements the training pipeline, including data preparation, model training, and K-fold cross-validation.
- **node_dataset.py**: Defines the dataset module with data loading and augmentation utilities.
- **node_net.py**: Defines the network module, including DNet, HDNet, and MHDNet architectures.
- **node_results.py**: Defines the results module with loss functions and evaluation metrics.
- **node_utils.py**: Provides utility functions for training and validation.

---

## 安装与依赖 / Installation and Dependencies

### 环境要求 / Environment Requirements
- Python 3.8 或以上版本 / Python 3.8 or higher
- PyTorch 1.10 或以上版本 / PyTorch 1.10 or higher
- 依赖库 / Required libraries:
  - `numpy`, `pandas`, `nibabel`, `scipy`, `scikit-learn`, `tabulate`

### 安装步骤 / Installation Steps
1. 克隆项目代码库 / Clone the project repository:
   ```bash
   git clone <repository_url>
   cd MHD_Nodet
   ```
2. 安装依赖 / Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. 确保数据目录（例如 `C:\Users\souray\Desktop\Tr`）包含 NIfTI 或 CSV 格式的医学影像数据 / Ensure the data directory (e.g., `C:\Users\souray\Desktop\Tr`) contains medical imaging data in NIfTI or CSV format.

---

## 使用方法 / Usage

1. **准备数据 / Prepare Data**：
   - 将医学影像数据（NIfTI 或 CSV 格式）放置在指定数据目录（例如 `C:\Users\souray\Desktop\Tr`）。
   - 确保文件名遵循 `case_<case_id>_<suffix>.nii.gz` 或 `case_<case_id>_<suffix>.csv` 的格式。

2. **配置超参数 / Configure Hyperparameters**：
   - 在 `node_train.py` 中调整超参数，例如 `batch_size`、`num_epochs`、`learning_rate` 和 `k_folds`。

3. **运行训练 / Run Training**：
   ```bash
   python node_pipline/node_train.py
   ```
   训练过程将自动进行 K 折交叉验证，并将模型权重和日志保存到指定目录（例如 `C:\Users\souray\Desktop\MHDNet0419`）。

4. **评估与导出 / Evaluation and Export**：
   - 使用 `node_results.py` 中的指标（Dice、IoU、Recall 等）评估模型性能。
   - 使用 `node_net.py` 中的 `run_example` 函数导出 ONNX 模型。

---

## 示例 / Example

以下是一个简单的运行示例，展示如何训练 MHDNet 模型并导出为 ONNX 格式：

The following is a simple example demonstrating how to train the MHDNet model and export it to ONNX format:

```python
# 在 node_train.py 中运行主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
```

```python
# 在 node_net.py 中运行示例以导出 ONNX 模型
from node_toolkit.node_net import example_mhdnet
example_mhdnet()
```

---

## 贡献 / Contributing

欢迎为 MHD_Nodet 项目贡献代码或提出建议！请按照以下步骤参与：

1. Fork 项目代码库 / Fork the project repository.
2. 创建特性分支 / Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. 提交更改 / Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. 推送至远程仓库 / Push to the remote repository:
   ```bash
   git push origin feature/your-feature
   ```
5. 提交 Pull Request / Submit a Pull Request.

---

## 联系方式 / Contact

- **作者 / Author**: Souray Meng (孟号丁)
- **邮箱 / Email**: souray@qq.com
- **机构 / Institution**: Tsinghua University (清华大学)

如有问题或建议，请通过邮箱联系或在项目仓库提交 Issue。

For questions or suggestions, please contact via email or submit an Issue on the project repository.

---

## 许可 / License

本项目采用 MIT 许可证，详情请见 `LICENSE` 文件。

This project is licensed under the MIT License. See the `LICENSE` file for details.


