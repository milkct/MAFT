from PIL import Image
import os


def convert_to_png(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件的后缀名是否为jpg
        if filename.endswith(".jpg"):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 打开jpg文件并转换为png
            img = Image.open(file_path)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_filepath = os.path.join(folder_path, png_filename)
            img.save(png_filepath, "PNG")

            # 可选：删除原始的jpg文件
            # os.remove(file_path)

            print(f"文件 {filename} 转换为 {png_filename}")


# 调用函数并传入文件夹路径
convert_to_png("E:\DATA\TWMARS\Train\masks")
