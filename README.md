 ```markdown
# MHD_Nodet Project

## Overview

The **MHD_Nodet** project (Multi Hypergraph Dynamic Node Network) is a sophisticated deep learning framework designed for processing complex medical imaging data, particularly for tasks such as plaque segmentation in medical images. Developed at Tsinghua University, this project leverages a hypergraph-based neural network architecture to model intricate data relationships, offering flexibility and robustness for 1D, 2D, and 3D data processing.

The framework is modular, comprising several key components:

- **net.py**: Defines the core neural network architecture, including `DNet`, `HDNet`, and `MHDNet` classes for dynamic hypergraph-based processing.
- **dataset.py**: Implements data loading and transformation utilities, including the `NodeDataset` class and augmentation techniques like normalization, rotation, flipping, shifting, and zooming.
- **train.py**: Contains the training pipeline, incorporating data preparation, model training, and k-fold cross-validation with a warmup cosine annealing learning rate scheduler.
- **results.py**: Provides loss functions (e.g., Dice, IoU, Focal, Lp) and evaluation metrics (e.g., recall, precision, F1, Dice, IoU) for regression, classification, and segmentation tasks.
- **utils.py**: Offers utility functions for training and validation, including data type mapping and logging.

## Features

- **Hypergraph-Based Architecture**: Utilizes `MHDNet` to model complex node and edge relationships, enabling dynamic processing across multiple subnetworks.
- **Flexible Dimensionality**: Supports 1D, 2D, and 3D data, making it adaptable to various medical imaging modalities.
- **Data Augmentation**: Includes robust augmentation techniques (e.g., `RandomRotate`, `RandomFlip`, `RandomShift`, `RandomZoom`) to enhance model generalization.
- **Comprehensive Loss and Metrics**: Supports multiple loss functions and metrics tailored for segmentation tasks, ensuring accurate model evaluation.
- **Cross-Validation**: Implements k-fold cross-validation for robust model training and validation.
- **ONNX Export**: Allows exporting trained models to ONNX format for deployment.

## Installation

To set up the MHD_Nodet project, ensure you have Python 3.8+ and the following dependencies installed:

```bash
pip install torch numpy pandas nibabel scipy sklearn
```

Clone the project repository:

```bash
git clone <repository_url>
cd mhd_nodet
```

## Usage

### Prepare Data

- Place your medical imaging data (e.g., `.nii.gz` or `.csv` files) in the specified `data_dir`.
- Ensure files follow the naming convention `case_<case_id>_<suffix>.<extension>`.

### Configure the Model

- Modify `train.py` to set hyperparameters (e.g., `batch_size`, `num_epochs`, `learning_rate`) and data paths (`data_dir`, `save_dir`).
- Adjust node and hyperedge configurations in `train.py` for specific tasks (e.g., plaque segmentation).

### Run Training

Execute the training script:

```bash
python train.py
```

The script performs k-fold cross-validation, saves the best model weights, and logs training metrics.

### Export Model

Use the `run_example` function in `net.py` to export the trained model to ONNX:

```python
example_mhdnet()
```

### Evaluate Results

Check the training logs (`training_log_fold<X>.json`) and model configurations (`model_config_fold<X>.json`) in the `save_dir` for performance metrics.

## Example

The `example_mhdnet` function in `net.py` demonstrates how to configure and run an `MHDNet` model with three subnetworks for a 3D segmentation task. It includes:

- **Subnetwork Configurations**: Defines node and hyperedge settings for processing 3D medical images.
- **Node Mapping**: Maps global nodes to subnetwork nodes for seamless data flow.
- **ONNX Export**: Exports the model to `MHDNet_example.onnx`.

To run the example:

```bash
python net.py
```

## Directory Structure

```
mhd_nodet/
├── node_pipline
│   └── node_train.py
├── node_toolkit
│   ├── node_dataset.py
│   ├── node_net.py
│   ├── node_results.py
│   └── node_utils.py
```

##
