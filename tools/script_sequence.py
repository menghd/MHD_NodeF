import os
import subprocess
import time
import sys

def replace_and_run(base_script_path, replace_map):
    """替换脚本中的关键词并运行"""
    # 检查基础脚本是否存在
    if not os.path.exists(base_script_path):
        print(f"错误：基础脚本不存在 - {base_script_path}")
        raise FileNotFoundError(f"找不到文件: {base_script_path}")

    # 尝试多种编码读取基础脚本内容
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    content = None
    for encoding in encodings:
        try:
            with open(base_script_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        error_msg = f"错误：无法用以下编码读取文件 {base_script_path}：{', '.join(encodings)}"
        print(error_msg)
        raise UnicodeDecodeError(error_msg)
    
    # 检查所有要替换的关键词是否存在
    missing_keys = [old_str for old_str in replace_map.keys() if old_str not in content]
    
    if missing_keys:
        print(f"警告：基础脚本中不存在以下关键词：{', '.join(missing_keys)}")
        # 询问用户是否继续
        while True:
            choice = input("是否跳过这些关键词继续运行？(y/n): ").strip().lower()
            if choice in ['y', 'n']:
                break
            print("请输入 'y' 或 'n'")
        
        if choice == 'n':
            error_msg = "用户选择中断操作"
            print(error_msg)
            raise ValueError(error_msg)

    # 创建临时脚本路径（改进命名方式，避免多层_temp累积）
    base_name = os.path.splitext(base_script_path)[0]
    # 移除可能存在的旧时间戳，避免文件名过长
    if '_temp_' in base_name:
        base_name = base_name[:base_name.rfind('_temp_')]
    temp_script_path = f"{base_name}_temp_{int(time.time())}.py"
    
    try:        
        # 执行替换
        new_content = content
        for old_str, new_str in replace_map.items():
            # 只替换存在的关键词
            if old_str in new_content:
                new_content = new_content.replace(old_str, new_str)
        
        # 写入临时脚本（使用与读取相同的编码）
        with open(temp_script_path, 'w', encoding=encoding) as f:
            f.write(new_content)
        
        # 运行临时脚本
        print(f"\n{'='*50}")
        print(f"正在运行替换后的脚本（替换项: {replace_map}）")
        print(f"{'='*50}")
        
        result = subprocess.run(
            ['python', temp_script_path],
            capture_output=True,
            text=True,
            encoding=encoding,
            errors='replace'
        )
        
        print("运行输出：")
        print(result.stdout)
        
        if result.stderr:
            print("警告信息：")
            print(result.stderr)
        
        time.sleep(1)
        
        return temp_script_path
            
    finally:
        # 清理当前生成的临时文件（如果运行失败）
        if 'result' not in locals() and os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
                print(f"已清理失败的临时文件: {temp_script_path}")
            except Exception as e:
                print(f"清理临时文件失败：{e}")

def process_script_sequence(original_script, replace_sequences):
    """处理单个脚本的替换序列"""
    if not os.path.exists(original_script):
        print(f"错误：未找到原始脚本 {original_script}")
        return False
    
    current_script = original_script
    temp_files = []  # 记录所有临时文件
    
    try:
        for i, replace_map in enumerate(replace_sequences):
            print(f"\n----- 开始第{i+1}轮替换 -----")
            try:
                new_script = replace_and_run(current_script, replace_map)
                # 如果是临时文件，添加到列表以便最后清理
                if '_temp_' in new_script:
                    temp_files.append(new_script)
                current_script = new_script
            except Exception as e:
                print(f"第{i+1}轮替换失败: {e}")
                return False
        
        return True
    
    finally:
        # 清理所有临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"已清理临时文件: {temp_file}")
                except Exception as e:
                    print(f"清理临时文件失败：{e}")

def main():
    # 定义脚本和对应的替换序列
    original_script1 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\node_pipline\test_UniConnNetI.py"
    original_script2 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\node_pipline\test_UniConnNetII.py"
    original_script3 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\node_pipline\test_UniConnNetIII.py"
    original_script4 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\try_train\17_restore_radsI.py"
    original_script5 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\try_train\17_restore_radsII.py"
    original_script6 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\try_train\17_restore_radsIII.py"
    original_script7 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\try_train\18_split_pred_cls.py"
    original_script8 = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\try_train\20_cls_val.py"
    
    # 定义替换序列1
    replace_sequences1 = [
        # test & restore
        {"fold1": "fold1"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"fold5": "fold1", "gt": "pred", "1004.nii.gz": "0004.nii.gz"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
    ]

    replace_sequences2 = [
        # test & restore
        {"fold1": "fold1"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"fold5": "fold1", "gt": "pred"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
    ]
    
    # 定义替换序列3
    replace_sequences3 = [
        {"fold1": "fold1"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"UniConnNetI": "UniConnNetII", "fold5": "fold1"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"UniConnNetII": "UniConnNetIII", "fold5": "fold1"},  
        {"fold1": "fold2"}, 
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"fold5_gt": "fold1_pred", "UniConnNetIII": "UniConnNetI"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"UniConnNetI": "UniConnNetII", "fold5": "fold1"},
        {"fold1": "fold2"},
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
        {"UniConnNetII": "UniConnNetIII", "fold5": "fold1"},  
        {"fold1": "fold2"}, 
        {"fold2": "fold3"},
        {"fold3": "fold4"},
        {"fold4": "fold5"},
    ]
    
    # 定义脚本与序列的组合
    script_sequence_pairs = [
        (original_script1, replace_sequences1),
        (original_script2, replace_sequences1),
        (original_script3, replace_sequences1),
        (original_script4, replace_sequences2),
        (original_script5, replace_sequences2),
        (original_script6, replace_sequences2),
        (original_script7, replace_sequences3),
        (original_script8, replace_sequences3),
    ]
    
    # 处理所有组合
    for i, (script, sequences) in enumerate(script_sequence_pairs, 1):
        print(f"\n{'#'*60}")
        print(f"开始处理第 {i} 个脚本: {os.path.abspath(script)}")
        print(f"{'#'*60}")
        
        if not process_script_sequence(script, sequences):
            print(f"第 {i} 个脚本处理失败，继续处理下一个...")
    
    print("\n所有任务已完成！")

if __name__ == "__main__":
    main()
