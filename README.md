# MHD_Nodet

**MHD_Nodet** (Multi-Hypergraph Dynamic Network) is a flexible deep learning framework designed for processing multi-modal medical imaging data using a hypergraph-based neural network architecture. It supports dynamic node connections, customizable data transformations, and multiple loss/metric functions for tasks such as segmentation and regression. This project is developed by Souray Meng at Tsinghua University.

## Features

- **Hypergraph-based Architecture**: Models complex relationships between data nodes using hyperedges.
- **Flexible Data Processing**: Supports various data formats (e.g., NIfTI, CSV) and transformations (e.g., rotation, normalization).
- **Customizable Loss and Metrics**: Includes Dice, IoU, Focal Loss, and more for segmentation and regression tasks.
- **Cross-Validation Support**: Implements K-fold cross-validation for robust model evaluation.
- **Open-Source**: Licensed under the MIT License.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/souraymeng/MHD_Nodet.git
   cd MHD_Nodet
