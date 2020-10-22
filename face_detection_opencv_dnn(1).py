from __future__ import division
import cv2
import tensorflow as tf
import numpy as np
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()  # 清除过往tensorflow数据记录
sess = tf.Session()
# sess.run(tf.global_variables_initializer())


def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    if len(bboxes)==0:
        print('抱歉，未检测到人脸')
    else:
        re=detect(bboxes,frame)
    return bboxes,re


def detect(bboxes, frame):
    data = []
    for i in bboxes:
        X = int(i[0] * 0.7)
        W = min(int(i[2] * 1.3), frame.shape[1])
        Y = int(i[1] * 0.7)
        H = min(int(i[3] * 1.3), frame.shape[0])
        img = cv2.resize(frame[Y:H, X:W], (W - X, H - Y))  # 裁剪后人脸
        img = cv2.resize(img, (100, 100))
        cv2.imshow("Face Detection Comparison", img)
        cv2.waitKey(0)

        img_final = np.asarray(img / 255, np.float32)
        data.append(img_final)

    result=gender_recognition(data)  # 性别分类
    return result


# 加载模型
def gender_recognition(data):
    result=[]
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + meta_graph)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))  # 加载最新模型到当前环境中
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}
        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)
        # print(classification_result) 输出预测矩阵
        # 根据索引通过字典对应的分类
        output = tf.argmax(classification_result, 1).eval()
        # print(output)

        for i in range(len(output)):
            print("第", i + 1, "张图预测:" + class_dict[output[i]])
            result.append(class_dict[output[i]])
        # print("可信度：" + output[0])
        return result




if __name__ == "__main__" :
    # 类别信息
    class_dict = {0: 'man', 1: 'woman'}

    # 最新模型地址
    model_path = 'C:/Users/Administrator/Desktop/VWebsite/CNN/model/'
    meta_graph = 'model-49-2020-03-11 13%3A20%3A20.meta'

    # 加载人脸检测器
    modelFile = "C:/Users/Administrator/Desktop/VWebsite/CNN/opencv_face_detector_uint8.pb"
    configFile = "C:/Users/Administrator/Desktop/VWebsite/CNN/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    conf_threshold = 0.7

    print('Loading...')
    # 输入图片
    # try:
    path = r'C:/Users/Administrator/Desktop/3.jpg'
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (200, 200))
    bboxes,re= detectFaceOpenCVDnn(net, frame)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    if len(bboxes)>0:
        for i,j in zip(bboxes,re):
            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), int(round(frameHeight / 150)), 8)
            cv2.putText(frame, str(j), (i[0], i[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    print(bboxes)
    cv2.imshow("Face Detection Comparison", frame)
    cv2.waitKey(0)
    # except:
    #     print('input error!')





