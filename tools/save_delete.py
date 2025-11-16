import os

def process_files(root_dir, keywords, mode='delete'):
    """
    处理文件夹中的文件，根据模式删除或保留文件
    :param root_dir: 根目录路径
    :param keywords: 关键词列表
    :param mode: 'delete' 删除包含关键词的文件, 'save' 删除不包含关键词的文件
    """
    # 初始化总匹配次数字典
    total_counts = {keyword: 0 for keyword in keywords}
    
    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            print(f"正在处理文件夹: {folder_name}")
            
            # 初始化当前文件夹匹配次数字典
            folder_counts = {keyword: 0 for keyword in keywords}
            
            # 遍历文件夹中的文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # 检查文件名是否包含关键词
                contains_keyword = False
                for keyword in keywords:
                    if keyword in file_name:
                        contains_keyword = True
                        folder_counts[keyword] += 1
                        total_counts[keyword] += 1
                        break
                
                # 根据模式决定是否删除文件
                if (mode == 'delete' and contains_keyword) or (mode == 'save' and not contains_keyword):
                    os.remove(file_path)
                    print(f"  已删除文件: {file_name}")
                else:
                    print(f"  已保留文件: {file_name}")
            
            # 打印当前文件夹的匹配次数
            print(f"文件夹 {folder_name} 的匹配次数：")
            for keyword, count in folder_counts.items():
                print(f"  关键词 '{keyword}': {count} 次")
    
    # 打印总匹配次数
    print("总匹配次数：")
    for keyword, count in total_counts.items():
        print(f"  关键词 '{keyword}': {count} 次")
    
    print(f"{mode} 模式操作完成！")

# 定义根目录
root_dir = r'C:\Users\PC\PycharmProjects\thu_xwh\Val_Data'

# 定义关键词列表
keywords = ['resnet18']  # 根据模式决定是删除还是保留的关键词

# 示例调用
# 删除模式（默认，删除包含关键词的文件）
process_files(root_dir, keywords, mode='delete')

# # 保留模式（删除不包含关键词的文件）
# process_files(root_dir, keywords, mode='save')
