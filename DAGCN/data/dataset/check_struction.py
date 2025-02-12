import os

def print_folder_structure(folder_path):
    """
    打印指定文件夹的结构。
    :param folder_path: 文件夹路径
    """
    print(f"文件夹结构: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)  # 计算当前文件夹的层级
        indent = ' ' * 4 * (level)  # 缩进
        print(f"{indent}{os.path.basename(root)}/")  # 输出文件夹
        subindent = ' ' * 4 * (level + 1)  # 子文件夹缩进
        for f in files:
            print(f"{subindent}{f}")  # 输出文件

# 检查cora_ml和citeseer文件夹结构
def check_folders_structure(root_dir):
    datasets = ["cora_ml", "citeseer", "chameleon", "squirrel", "directed_roman_empire"]  # 要检查的文件夹名称
    for dataset_name in datasets:
        dataset_path = os.path.join(root_dir, dataset_name)
        print(f"正在检查：{dataset_path}")  # 打印检查的路径
        if os.path.exists(dataset_path):
            print_folder_structure(dataset_path)
        else:
            print(f"{dataset_name} 文件夹不存在！")

# 主函数
def main():
    root_dir = os.getcwd()  # 获取当前工作目录
    print(f"当前工作目录: {root_dir}")  # 打印当前工作目录
    check_folders_structure(root_dir)

if __name__ == "__main__":
    main()
