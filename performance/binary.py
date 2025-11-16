import numpy as np

def calculate_binary_metrics(f1_resnet34_fold5_matrix, positive_classes):
    """计算二分类混淆矩阵及指标，指定正类类别"""
    num_classes = len(f1_resnet34_fold5_matrix)
    positive_classes = set(str(x) for x in positive_classes)  # 转换为字符串集合
    cm = np.array(f1_resnet34_fold5_matrix)
    
    # 初始化二分类混淆矩阵
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    # 遍历混淆矩阵，重新分配到二分类
    for i in range(num_classes):
        for j in range(num_classes):
            i_str, j_str = str(i), str(j)
            if i_str in positive_classes and j_str in positive_classes:
                TP += cm[i][j]  # 正类预测为正类
            elif i_str not in positive_classes and j_str not in positive_classes:
                TN += cm[i][j]  # 负类预测为负类
            elif i_str in positive_classes and j_str not in positive_classes:
                FN += cm[i][j]  # 正类预测为负类
            else:
                FP += cm[i][j]  # 负类预测为正类
    
    # 计算指标
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    
    # 计算MCC
    mcc_numerator = TP * TN - FP * FN
    mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-8)
    mcc = mcc_numerator / (mcc_denominator + 1e-8)
    
    # 格式化百分比，保留两位小数
    metrics = {
        'binary_f1_resnet34_fold5_matrix': [[TN, FP], [FN, TP]],
        'accuracy': f"{accuracy * 100:.2f}%",
        'precision': f"{precision * 100:.2f}%",
        'recall': f"{recall * 100:.2f}%",
        'specificity': f"{specificity * 100:.2f}%",
        'f1': f"{f1 * 100:.2f}%",
        'mcc': f"{mcc * 100:.2f}%"
    }
    
    return metrics

if __name__ == "__main__":
    # 示例混淆矩阵
    f1_resnet34_fold5_matrix = [
    [
      8,
      7,
      4,
      4
    ],
    [
      5,
      13,
      6,
      0
    ],
    [
      1,
      1,
      35,
      9
    ],
    [
      1,
      0,
      12,
      61
    ]
  ]
    
    # 指定正类类别，使用集合
    positive_classes = {3,2}  # 正类为 {3, 2}，负类为 {0, 1}
    # positive_classes = {3}   # 或者只指定 {3} 为正类，负类为 {0, 1, 2}
    
    # 计算二分类指标
    results = calculate_binary_metrics(f1_resnet34_fold5_matrix, positive_classes)
    
    print("二分类混淆矩阵:")
    print(np.array(results['binary_f1_resnet34_fold5_matrix']))
    print("指标:", results)
