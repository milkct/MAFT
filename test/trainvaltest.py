
from sklearn.model_selection import train_test_split
import os

imagedir = r"E:\DATA\TWMARS\Test\masks"
outdir = r'E:\DATA\TWMARS'
os.makedirs(outdir,exist_ok=True)

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    images.append(filename)

# Split the data into training, validation, and test sets (8:1:1 ratio)
train_size = 0.7
val_size = 0.2
test_size = 0.1

train, temp = train_test_split(images, test_size=(val_size + test_size), random_state=0)
val, test = train_test_split(temp, test_size=(test_size / (val_size + test_size)), random_state=0)

# Write the lists to text files
with open(os.path.join(outdir, "train.txt"), 'w') as f:
    f.write('\n'.join(train))

with open(os.path.join(outdir, "val.txt"), 'w') as f:
    f.write('\n'.join(val))

with open(os.path.join(outdir, "test.txt"), 'w') as f:
    f.write('\n'.join(test))
#无测试集，以下

# import os
# import random
# from sklearn.model_selection import train_test_split
#
# input_dir = r'E:\DATA\TWMARS\Train\imgs'
# output_dir= r'E:\DATA\TWMARS'
#
#
# # Get list of image filenames in input directory
# image_filenames = [filename for filename in os.listdir(input_dir) if filename.endswith(".jpg")]
#
# # Specify proportion of dataset to be used for validation
# validation_ratio = 0.2
#
# # Split dataset into training and validation sets
# train_filenames, val_filenames = train_test_split(image_filenames, test_size=validation_ratio)
#
# # Create output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Write training set filenames to file
# train_file_path = os.path.join(output_dir, "train.txt")
# with open(train_file_path, "w") as file:
#     for filename in train_filenames:
#         file.write(os.path.splitext(filename)[0] + "\n")
# print(f"Training set file saved to {train_file_path}")
#
# # Write validation set filenames to file
# val_file_path = os.path.join(output_dir, "val.txt")
# with open(val_file_path, "w") as file:
#     for filename in val_filenames:
#         file.write(os.path.splitext(filename)[0] + "\n")
# print(f"Validation set file saved to {val_file_path}")

#只有测试集
# import os
#
# # 文件夹路径
# folder_path = "E:\DATA\TWMARS\Test\masks"
#
# # 获取文件夹中所有文件名
# file_names = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith(".jpg") or file.endswith(".png")]
#
# # 写入文本文件
# with open(r"E:\DATA\TWMARS\Test\test.txt", "w") as f:
#     for file_name in file_names:
#         f.write(file_name + "\n")

