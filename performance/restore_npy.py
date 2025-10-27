"""
MHD_Nodet Project - Postprocessing Module
==========================================
This module provides functions to process prediction results from the MHD_Nodet project, generating a JSON file with per-patient R/L side classifications.
- Supports processing arbitrary .npy prediction files with flexible number of classes.
- Maps prediction indices to specific categories (0->1, 1->2, 2->3, 3->4).
- Aggregates predictions by patient ID and R/L side, using weighted sum within ROIs (weights normalized to sum to 1) and max category across ROIs.
- Saves results to a JSON file, indicating the source .npy file suffixes.

项目：MHD_Nodet - 后处理模块
本模块提供函数来处理 MHD_Nodet 项目的预测结果，生成按患者 ID 和 R/L 侧分类的 JSON 文件。
- 支持处理任意 .npy 预测文件，适应灵活的分类数量。
- 将预测索引映射到特定类别（0->1, 1->2, 2->3, 3->4）。
- 按患者 ID 和 R/L 侧分类聚合结果，在 ROI 内部使用加权求和（权重归一化后相加为1），ROI 间取最大类别。
- 将结果保存为 JSON 文件，标明源 .npy 文件后缀。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import os
import json
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Category mapping dictionary
CATEGORY_MAPPING = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
}

def load_mappings(case_mapping_path, roi_mapping_path):
    """
    Load case-to-ROI and ROI-to-side mapping files.

    加载 case 到 ROI 和 ROI 到 R/L 侧的映射文件。

    Args:
        case_mapping_path (str): Path to case_to_roi_mapping.json
        roi_mapping_path (str): Path to roi_mapping.json

    Returns:
        tuple: (case_to_roi, roi_to_side) dictionaries
    """
    try:
        with open(case_mapping_path, 'r') as f:
            case_to_roi = json.load(f)
        with open(roi_mapping_path, 'r') as f:
            roi_to_side = json.load(f)
        logger.info(f"Loaded mappings: {case_mapping_path}, {roi_mapping_path}")
        return case_to_roi, roi_to_side
    except Exception as e:
        logger.error(f"Error loading mappings: {e}")
        raise

def get_prediction_category(npy_path):
    """
    Load .npy prediction file and return the raw softmax probabilities.

    加载 .npy 预测文件，返回原始 softmax 概率。

    Args:
        npy_path (str): Path to .npy file

    Returns:
        np.ndarray: Raw softmax probabilities, or None if file is missing/invalid
    """
    try:
        if not os.path.exists(npy_path):
            logger.warning(f"Prediction file missing: {npy_path}")
            return None
        pred = np.load(npy_path)
        num_classes = pred.shape[0]
        if num_classes < 1:
            logger.error(f"Invalid prediction shape {pred.shape} in {npy_path}, expected at least 1 channel")
            return None
        logger.debug(f"Loaded {npy_path}, num_classes: {num_classes}")
        return pred
    except Exception as e:
        logger.error(f"Error processing {npy_path}: {e}")
        return None

def process_predictions(data_dir, npy_suffixes, weights, case_to_roi, roi_to_side):
    """
    Process .npy prediction files with specified suffixes and weights, aggregate by patient ID and R/L side.

    处理指定后缀和权重的 .npy 预测文件，按患者 ID 和 R/L 侧聚合。

    Args:
        data_dir (str): Directory containing .npy prediction files
        npy_suffixes (list): List of suffixes of .npy files to process (e.g., ['0015.npy', '1015.npy'])
        weights (list): List of weights corresponding to npy_suffixes, normalized to sum to 1
        case_to_roi (dict): Mapping of case ID to patient_id_roi
        roi_to_side (dict): Mapping of patient ID to ROI-to-side

    Returns:
        dict: Aggregated predictions {patient_id: {"R": category, "L": category}}
    """
    # Normalize weights
    weights = np.array(weights, dtype=float)
    if len(weights) != len(npy_suffixes):
        logger.error(f"Weights length {len(weights)} does not match npy_suffixes length {len(npy_suffixes)}")
        raise ValueError("Weights and npy_suffixes must have the same length")
    weights = weights / np.sum(weights)
    logger.info(f"Normalized weights: {weights}")

    results = defaultdict(lambda: defaultdict(lambda: {"R": [], "L": []}))  # Nested defaultdict for patient_id -> roi -> side
    all_files = sorted(os.listdir(data_dir))

    # Collect raw predictions
    for file in all_files:
        for suffix in npy_suffixes:
            if file.endswith(f'_{suffix}'):
                # Extract case_id as 'case_XXXX' to match case_to_roi keys
                parts = file.split('_')
                if len(parts) < 3 or parts[0] != 'case':
                    logger.warning(f"Invalid filename format: {file}, skipping")
                    continue
                case_id = f"case_{parts[1]}"  # e.g., 'case_0000'
                npy_path = os.path.join(data_dir, file)
                pred = get_prediction_category(npy_path)
                
                if pred is None:
                    continue

                if case_id not in case_to_roi:
                    logger.warning(f"Case ID {case_id} not found in case_to_roi_mapping")
                    continue

                patient_roi = case_to_roi[case_id]  # e.g., "1_roi11"
                try:
                    patient_id, roi = patient_roi.split('_')  # e.g., "1", "roi11"
                except ValueError:
                    logger.warning(f"Invalid patient_roi format: {patient_roi}")
                    continue

                if patient_id not in roi_to_side:
                    logger.warning(f"Patient ID {patient_id} not found in roi_to_side")
                    continue
                if roi not in roi_to_side[patient_id]:
                    logger.warning(f"ROI {roi} not found for patient {patient_id}")
                    continue

                side = roi_to_side[patient_id][roi]  # e.g., "L"
                if side not in ["R", "L"]:
                    logger.warning(f"Invalid side {side} for {patient_roi}")
                    continue

                # Store raw prediction probabilities
                results[patient_id][roi][side].append((pred, npy_suffixes.index(suffix)))
                logger.debug(f"Added prediction for patient {patient_id}, ROI {roi}, side {side}, suffix {suffix}")

    # Aggregate by weighted sum within each ROI and max across ROIs
    final_results = {}
    for patient_id, rois in results.items():
        final_results[patient_id] = {"R": [], "L": []}
        for roi, sides in rois.items():
            for side in ["R", "L"]:
                if sides[side]:
                    # Weighted sum of predictions within ROI
                    weighted_pred = None
                    for pred, suffix_idx in sides[side]:
                        weight = weights[suffix_idx]
                        if weighted_pred is None:
                            weighted_pred = weight * pred
                        else:
                            weighted_pred += weight * pred
                    # Take argmax to get category
                    category_idx = np.argmax(weighted_pred, axis=0).flatten()[0]
                    if category_idx not in CATEGORY_MAPPING:
                        logger.error(f"Invalid category index {category_idx} for patient {patient_id}, ROI {roi}, side {side}")
                        continue
                    category = CATEGORY_MAPPING[category_idx]
                    final_results[patient_id][side].append(category)
                    logger.debug(f"ROI {roi} for patient {patient_id}, side {side}: weighted category {category}")

        # Aggregate across ROIs by taking max category
        for side in ["R", "L"]:
            if final_results[patient_id][side]:
                # Sort categories by their index in CATEGORY_MAPPING to get the max
                final_results[patient_id][side] = sorted(
                    final_results[patient_id][side],
                    key=lambda x: max([k for k, v in CATEGORY_MAPPING.items() if v == x]),
                    reverse=True
                )[0]
            else:
                final_results[patient_id][side] = None
            logger.debug(f"Patient {patient_id}, side {side}: final category {final_results[patient_id][side]}")

    return final_results

def save_results(results, npy_suffixes, output_path):
    """
    Save aggregated predictions to JSON file, including source .npy suffixes.

    将聚合的预测结果保存为 JSON 文件，包含源 .npy 文件后缀。

    Args:
        results (dict): Aggregated predictions
        npy_suffixes (list): List of suffixes of processed .npy files
        output_path (str): Path to save JSON file
    """
    try:
        output_data = {
            "source_npy_suffix": npy_suffixes,
            "predictions": results
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Saved predictions to {output_path} with source_npy_suffix: {npy_suffixes}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        raise

def postprocess_npy_predictions(data_dir, npy_suffixes, weights, case_mapping_path, roi_mapping_path, output_path):
    """
    Main function to postprocess .npy predictions and generate per-patient R/L side classifications.

    后处理 .npy 预测结果的主函数，生成按患者 R/L 侧的分类结果。

    Args:
        data_dir (str): Directory containing .npy prediction files
        npy_suffixes (list): List of suffixes of .npy files to process (e.g., ['0015.npy', '1015.npy'])
        weights (list): List of weights corresponding to npy_suffixes
        case_mapping_path (str): Path to case_to_roi_mapping.json
        roi_mapping_path (str): Path to roi_mapping.json
        output_path (str): Path to save output JSON file
    """
    # Load mappings
    case_to_roi, roi_to_side = load_mappings(case_mapping_path, roi_mapping_path)

    # Process predictions
    results = process_predictions(data_dir, npy_suffixes, weights, case_to_roi, roi_to_side)

    # Save results
    save_results(results, npy_suffixes, output_path)

def main():
    """
    Example usage to process multiple .npy files with specified suffixes and weights.

    示例用法，处理指定后缀和权重的多个 .npy 文件。
    """
    # Paths
    data_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\scratch_imagesTs"
    case_mapping_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\ValNiigzROI\case_to_roi_mapping.json"
    roi_mapping_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\ValNiigzROI\roi_mapping.json"
    output_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\ValNiigzData\unicom0_fold1_pred_Feature_1.json"
    #output_path = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\ValNiigzData\f1_resnet34_fold5_pred_Feature_1.json"

    # Process multiple .npy files with weights
    #npy_suffixes = ["f1_resnet34_fold5_pred_5015.npy"]
    #weights = [1]
    npy_suffixes = ["unicom0_fold1_pred_0015.npy", "unicom0_fold1_pred_1015.npy", "unicom0_fold1_pred_2015.npy", "unicom0_fold1_pred_3015.npy", "unicom0_fold1_pred_4015.npy"]
    weights = [1, 1, 1, 1, 1]
    postprocess_npy_predictions(data_dir, npy_suffixes, weights, case_mapping_path, roi_mapping_path, output_path)

if __name__ == "__main__":
    main()
