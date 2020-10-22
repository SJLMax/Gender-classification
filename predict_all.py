from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

x_test = np.load('/home/ugrad/Shang/CNN/data_new/x_test_100.npy')
y_test = np.load('/home/ugrad/Shang/CNN/data_new/y_test_100.npy')


# 预测
def prediction(data):
    with tf.Session() as sess:
        model_path = '/home/ugrad/Shang/CNN/model/model_new/'
        saver = tf.train.import_meta_graph(model_path + 'model-49-2020-03-11 13:20:20.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))  # 加载最新模型到当前环境中
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}
        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)

        # 根据索引通过字典对应的分类
        output = tf.argmax(classification_result, 1).eval()
    return output

tf.reset_default_graph()  # 清除过往tensorflow数据记录
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_predict = prediction(x_test)


# 混淆矩阵绘制
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(2)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/home/ugrad/Shang/CNN/result/Confusion_Matrix_new.jpg')


confusion_matrix = tf.contrib.metrics.confusion_matrix(y_test, y_predict, num_classes=None, dtype=tf.int32, name=None, weights=None)
confusion_matrix = sess.run(confusion_matrix)
plot_confusion_matrix(confusion_matrix)

# 评价指标
accu = [0, 0]
column = [0, 0]
line = [0, 0]
recall = [0, 0]  # 召回率
precision = [0, 0]  # 精准率
accuracy = 0  # 准确率
Macro_P = 0  # 宏查准率（宏精准率）
Macro_R = 0  # 宏查全率（宏召回率）
# 准确率
for i in range(0, 2):
    accu[i] = confusion_matrix[i][i]
    accuracy += float(accu[i]) / len(y_test)

print('accuracy:')
print(accuracy)
# 宏召回率
for i in range(0, 2):
    for j in range(0, 2):
        column[i] += confusion_matrix[j][i]
    if column[i] != 0:
        recall[i] = float(accu[i]) / column[i]
Macro_R = np.array(recall).mean()

# 宏精准率
for i in range(0, 2):
    for j in range(0, 2):
        line[i] += confusion_matrix[i][j]
    if line[i] != 0:
        precision[i] = float(accu[i]) / line[i]
Macro_P = np.array(precision).mean()

# 宏F1
Macro_F1 = (2 * (Macro_P * Macro_R)) / (Macro_P + Macro_R)
print("宏P、R、F1：")
print(Macro_P, Macro_R, Macro_F1)