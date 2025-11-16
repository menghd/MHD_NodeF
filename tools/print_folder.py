import os

def print_directory_structure(root_directory, indent="", max_depth=float('inf'), current_depth=0, save=False, output_file="directory_structure.txt"):
    """
    打印指定路径下的文件和文件夹结构，并可选保存到文件。

    Args:
        root_directory (str): 根目录路径
        indent (str): 当前的缩进符号，用于显示层次结构
        max_depth (int or float): 最大递归深度，默认为无穷大（递归所有子目录）
        current_depth (int): 当前递归深度，用于控制递归次数
        save (bool): 是否将输出保存到文件，默认为 False
        output_file (str): 保存输出时的文件名，默认为 'directory_structure.txt'
    """
    # 如果当前深度超过最大深度，则停止递归
    if current_depth > max_depth:
        return

    # 如果需要保存，准备一个列表来存储输出行
    output_lines = [] if save else None

    # 遍历根目录下的所有文件和文件夹
    for item in os.listdir(root_directory):
        item_path = os.path.join(root_directory, item)

        # 构建当前行的输出
        line = indent + "|-- " + item
        print(line)
        if save:
            output_lines.append(line)

        # 如果是文件夹，则递归调用自己
        if os.path.isdir(item_path):
            # 增加缩进，递归打印子文件夹，并增加当前深度
            if save:
                # 递归时传递 output_lines 以收集所有输出
                sub_lines = print_directory_structure(item_path, indent + "    ", max_depth, current_depth + 1, save, output_file)
                if sub_lines:
                    output_lines.extend(sub_lines)
            else:
                print_directory_structure(item_path, indent + "    ", max_depth, current_depth + 1, save, output_file)

    # 如果 save=True 且是根调用（current_depth=0），将内容写入文件
    if save and current_depth == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"文件架构图 ({root_directory}):\n")
            for line in output_lines:
                f.write(line + '\n')
        print(f"文件结构已保存到 {output_file}")

    # 返回 output_lines 以供递归调用使用
    return output_lines if save else None


# 指定要打印的路径
root_directory = r'C:\Users\PC\PycharmProjects\thu_xwh\Val_Data\ValNiigzData\1'

# 打印文件架构图并保存
print(f"文件架构图 ({root_directory}):")
print_directory_structure(root_directory, max_depth=1, save=True)  # 设置 save=True 保存输出
