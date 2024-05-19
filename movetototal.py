import os
import shutil

# 源文件夹路径
source_folder = 'ttf'

# 目标文件夹路径
target_folder = os.path.join(source_folder, 'total')

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹
for root, dirs, files in os.walk(source_folder):
    for dir_name in dirs:
        # 检查文件夹名称是否以'_png'结尾
        if dir_name.endswith('_png'):
            folder_path = os.path.join(root, dir_name)
            # 遍历文件夹中的文件
            for filename in os.listdir(folder_path):
                source_file_path = os.path.join(folder_path, filename)
                target_file_path = os.path.join(target_folder, filename)
                # 复制文件到目标文件夹中
                shutil.copy2(source_file_path, target_file_path)
                print(f"Copied: {filename} from {folder_path} to {target_folder}")
