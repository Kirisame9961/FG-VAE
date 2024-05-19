import os
from fontTools.ttLib import TTFont

# 设置常量
TTF_FOLDER = "ttf"  # 存放.ttf文件的文件夹
FONT_SIZE = 100

# 获取所有.ttf文件
ttf_files = [f for f in os.listdir(TTF_FOLDER) if f.endswith('.TTF')]

# 遍历每个.ttf文件
for TTF in ttf_files:
    TTF_PATH = os.path.join(TTF_FOLDER, TTF)
    ttf = TTFont(TTF_PATH, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)

    # 创建存储字符文本文件的文件夹
    txt_folder = os.path.join("ttf", TTF.split(".")[0])
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    # 读取字体文件并生成对应的文本文件
    for x in ttf["cmap"].tables:
        for y in x.cmap.items():
            try:
                char_unicode = chr(y[0])
                char_utf8 = char_unicode.encode('utf_8')
                char_name = y[1]
                with open(os.path.join(txt_folder, char_name + '.txt'), 'wb') as f:
                    f.write(char_utf8)
            except Exception as e:
                print(f"Error creating text file: {str(e)}")
                pass
    ttf.close()

    # 创建存储生成图片的文件夹
    png_folder = os.path.join("ttf", f"{TTF.split('.')[0]}_png")
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 生成PNG图片
    files = os.listdir(txt_folder)
    for filename in files:
        try:
            name, ext = os.path.splitext(filename)
            input_txt = os.path.join(txt_folder, filename)
            output_png = os.path.join(png_folder, name + "_" + str(TTF.split('.')[0]) + ".png")

            with open(input_txt, "r", encoding='utf-8') as fileObject:
                data = fileObject.read()

            cmd = f"convert -font {TTF_PATH} -pointsize {FONT_SIZE} label:{data} {output_png}"
            os.system(cmd)
            print(f"PNG file {output_png} created.")
        except Exception as e:
            print(f"Error creating PNG file: {str(e)}")
            pass

    print(f"Finished generating PNG files for {TTF}.")
