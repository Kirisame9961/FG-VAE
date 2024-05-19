import os
import re

# 指定根文件夹路径
root_folder = 'ttf'

# 定义正则表达式，匹配文件名中 uniXXXX 形式的编号
pattern = re.compile(r'uni([0-9A-Fa-f]+)')

# 遍历根文件夹及其子文件夹
for root, dirs, files in os.walk(root_folder):
    for dir_name in dirs:
        # 检查文件夹名称是否以'_png'结尾
        if dir_name.endswith('_png'):
            folder_path = os.path.join(root, dir_name)
            # 遍历文件夹中的文件
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # 检查文件名是否匹配指定的模式并且以"uni"开头
                match = pattern.match(filename)
                if match and filename.startswith("uni"):
                    hex_number = match.group(1)

                    # 将十六进制字符串转换为整数
                    dec_number = int(hex_number, 16)

                    # 删除不在范围内的文件
                    if dec_number < 0x4e00 or dec_number > 0x9fff:
                        os.remove(file_path)
                        print(f"Deleted: {filename}")
                    else:
                        print(f"Kept: {filename}")
                else:
                    os.remove(file_path)
                    print(f"Deleted: {filename} (doesn't match pattern or doesn't start with 'uni')")
