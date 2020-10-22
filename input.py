import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# 数据集地址
path = '/home/ugrad/Shang/CNN/data_new/'
# path = 'C:/Users/lenovo/Desktop/tensor+CNN_multi/CNN/test_pic/'

# 将所有的图片resize
w = 100
h = 100
c = 3

# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    print(cate)
    imgs = []
    labels = []
    print('读取图像开始')
    for id, folder in enumerate(cate):
        folder1 = folder.split('/')
        idx = folder1[6]
        print(idx, folder)
        for im in glob.glob(folder + '/*.jpg'):
            try:
                img = cv2.imread(im)
                img = cv2.resize(img, (w, h))
                img = img / 255
                imgs.append(img)
                labels.append(int(idx))
            except:
                print('error')
    print('读取图像结束')
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
# 打乱顺序
num_example = data.shape[0]   # 图片总数
print(num_example)
arr = np.arange(num_example)  # 产生一个num_example范围，步长为1的序列
label_temp = label[arr]
# print(label_temp)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# 将所有数据分为训练集/验证集/测试集
ratio = 0.8
s = np.int(num_example * ratio)
s1 = np.int(num_example * 0.9)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:s1]
y_val = label[s:s1]
x_test = data[s1:]
y_test = label[s1:]
print(np.shape(x_test))


np.save('/home/ugrad/Shang/CNN/data_new/x_train_100', x_train)
np.save('/home/ugrad/Shang/CNN/data_new/y_train_100', y_train)
np.save('/home/ugrad/Shang/CNN/data_new/x_val_100', x_val)
np.save('/home/ugrad/Shang/CNN/data_new/y_val_100', y_val)
np.save('/home/ugrad/Shang/CNN/data_new/x_test_100', x_test)
np.save('/home/ugrad/Shang/CNN/data_new/y_test_100', y_test)

