import os
import shutil
import pandas as pd

def reorganize_data(input_root, output_dir, mode='2Tr'):
    """
    数据格式转换函数
    参数:
        input_root: 输入数据根目录
        output_dir: 输出目录
        mode: '2Tr' 表示正向转换到 Tr 格式, 'Tr2' 表示逆向恢复到原始格式
    """
    if mode == '2Tr':
        # 正向转换：从原始格式到 Tr 格式
        images_tr_dir = os.path.join(input_root, 'imagesTr')
        labels_tr_dir = os.path.join(input_root, 'labelsTr')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有 case 的编号
        case_ids = set()
        for csv_file in os.listdir(labels_tr_dir):
            if csv_file.endswith('.csv'):
                case_id = csv_file.split('_')[1].split('.')[0]
                case_ids.add(case_id)
        
        for case_id in sorted(case_ids):
            # 处理 NIfTI 文件（0000 到 0004）
            for suffix in ['0000', '0001', '0002', '0003', '0004']:
                nii_file = f'case_{case_id}_{suffix}.nii.gz'
                src_nii_path = os.path.join(images_tr_dir, nii_file)
                dst_nii_path = os.path.join(output_dir, nii_file)
                
                if os.path.exists(src_nii_path):
                    shutil.copy2(src_nii_path, dst_nii_path)
                    print(f'复制 NIfTI 文件: {nii_file}')
                else:
                    print(f'未找到 NIfTI 文件: {nii_file}')
            
            # 处理 CSV 文件
            csv_file = f'case_{case_id}.csv'
            csv_path = os.path.join(labels_tr_dir, csv_file)
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                features = [f'Feature_{i}' for i in range(1, 10)]
                if all(feature in df.columns for feature in features):
                    for i, feature in enumerate(features, start=5):
                        output_csv = f'case_{case_id}_{i:04d}.csv'
                        output_csv_path = os.path.join(output_dir, output_csv)
                        feature_df = df[[feature]].copy()
                        feature_df.columns = ['Value']
                        feature_df.to_csv(output_csv_path, index=False)
                        print(f'生成 CSV 文件: {output_csv}')
                else:
                    print(f'CSV 文件 {csv_file} 缺少部分 Feature 列')
            else:
                print(f'未找到 CSV 文件: {csv_file}')
    
    elif mode == 'Tr2':
        # 逆向转换：从 Tr 格式恢复到原始格式
        images_tr_dir = os.path.join(output_dir, 'imagesTr')
        labels_tr_dir = os.path.join(output_dir, 'labelsTr')
        
        # 创建输出目录
        os.makedirs(images_tr_dir, exist_ok=True)
        os.makedirs(labels_tr_dir, exist_ok=True)
        
        # 获取所有 case 的编号
        case_ids = set()
        for file in os.listdir(input_root):
            if file.startswith('case_') and file.endswith('.csv'):
                case_id = file.split('_')[1]
                case_ids.add(case_id)
        
        for case_id in sorted(case_ids):
            # 处理 NIfTI 文件（0000 到 0004）
            for suffix in ['0000', '0001', '0002', '0003', '0004']:
                nii_file = f'case_{case_id}_{suffix}.nii.gz'
                src_nii_path = os.path.join(input_root, nii_file)
                dst_nii_path = os.path.join(images_tr_dir, nii_file)
                
                if os.path.exists(src_nii_path):
                    shutil.copy2(src_nii_path, dst_nii_path)
                    print(f'恢复 NIfTI 文件: {nii_file}')
                else:
                    print(f'未找到 NIfTI 文件: {nii_file}')
            
            # 处理 CSV 文件（0005 到 0012）
            feature_dfs = []
            for i in range(5, 14):
                csv_file = f'case_{case_id}_{i:04d}.csv'
                csv_path = os.path.join(input_root, csv_file)
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if 'Value' in df.columns:
                        feature_dfs.append(df['Value'].rename(f'Feature_{i-4}'))
                    else:
                        print(f'CSV 文件 {csv_file} 格式错误，缺少 Value 列')
                else:
                    print(f'未找到 CSV 文件: {csv_file}')
            
            # 合并所有 Feature 列
            if feature_dfs:
                combined_df = pd.concat(feature_dfs, axis=1)
                output_csv = f'case_{case_id}.csv'
                output_csv_path = os.path.join(labels_tr_dir, output_csv)
                combined_df.to_csv(output_csv_path, index=False)
                print(f'恢复 CSV 文件: {output_csv}')
            else:
                print(f'无法为 case_{case_id} 恢复 CSV 文件，缺少 Feature 数据')

    else:
        raise ValueError("mode 参数必须为 '2Tr' 或 'Tr2'")

if __name__ == '__main__':
    # 设置路径
    input_root = r'C:\Users\PC\PycharmProjects\thu_xwh\Data\TrainNiigzCsvData'
    output_dir = os.path.join(input_root, 'Tr')
    
    # 正向转换示例
    print("执行正向转换 (2Tr):")
    reorganize_data(input_root, output_dir, mode='2Tr')
    
    # # 逆向转换示例
    # print("\n执行逆向转换 (Tr2):")
    # reorganize_data(output_dir, input_root, mode='Tr2')
