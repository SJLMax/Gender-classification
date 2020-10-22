import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 清除过往tensorflow数据记录
tf.reset_default_graph()
# 模型保存地址
model_path = '/home/ugrad/Shang/CNN/model/model_new/'
logs_train_dir = "/home/ugrad/Shang/CNN/logs/logs_new/"

# 将所有的图片resize成64*64
w = 100
h = 100
c = 3

x_train = np.load('/home/ugrad/Shang/CNN/data_new/x_train_100.npy')
y_train = np.load('/home/ugrad/Shang/CNN/data_new/y_train_100.npy')
x_val = np.load('/home/ugrad/Shang/CNN/data_new/x_val_100.npy')
y_val = np.load('/home/ugrad/Shang/CNN/data_new/y_val_100.npy')



# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [5, 5, 3, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6 * 6 * 128
        reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


# ---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    with tf.name_scope('acc'):
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', acc)


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# --------------训练和测试数据，可将n_epoch设置更大一些-------------
n_epoch = 30
batch_size = 64

merged = tf.summary.merge_all()  # 合并所有的scalar

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  # 保存到log里面
train_writer1 = tf.summary.FileWriter(logs_train_dir)  # 第二个log不需要sess.graph

print('网络开始训练时间')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

ValAcc=[]
TrainAcc=[]
ValLoss=[]
TrainLoss=[]
times=[]

for epoch in range(n_epoch):
    start_time = time.time()
    print("step:\t%d" % epoch)

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_step, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1

        summary_tmp = sess.run(merged, feed_dict={x: x_train_a, y_: y_train_a})  # 开始运行tensorboard
        train_writer.add_summary(summary=summary_tmp, global_step=epoch)  # 将数值写入

    lossF = sess.run(loss, feed_dict={x: x_train_a, y_: y_train_a})  # 这个就是要看的train loss

    training_loss = np.sum(train_loss) / n_batch
    training_acc = np.sum(train_acc) / n_batch
    print("   train loss: %f" % (training_loss))
    print("   train acc: %f" % (training_acc))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
        # print(logits)

        sum_tmp = sess.run(merged, feed_dict={x: x_val_a, y_: y_val_a})  # 开始运行
        train_writer1.add_summary(summary=sum_tmp, global_step=epoch)  # 将数值写入

    accuracy = sess.run(acc, feed_dict={x: x_val_a, y_: y_val_a})  # val acc就是这个了

    validation_loss = np.sum(val_loss) / n_batch
    validation_acc = np.sum(val_acc) / n_batch
    print("   validation loss: %f" % (validation_loss))
    print("   validation acc: %f" % (validation_acc))

    ValAcc.append(validation_acc)   # 扩展列表
    TrainAcc.append(training_acc)
    ValLoss.append(validation_loss)
    TrainLoss.append(training_loss)
    times.append(epoch)   # 扩展列表

timeArray = time.localtime()
now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)  # 时间
print('网络结束时间')
print(now)
# saver.save(sess, model_path)
saver.save(sess, model_path+'model-' + str(epoch)+'-'+now)
train_writer.close()
# train_writer1.close()
sess.close()


# -------------绘制Loss和Acc曲线------------------
def map(label, x1, x2, p):
    plt.figure()
    plt.plot(np.arange(n_epoch), x1, 'r', label='Train')
    plt.plot(np.arange(n_epoch), x2, 'b', label='Val')
    # plt.xlim((0, n_epoch))
    plt.ylim((0, p))
    plt.xlabel('Iterations',fontsize=18)
    plt.ylabel(label, fontsize=18)
    plt.legend()
    plt.savefig('/home/ugrad/Shang/CNN/map/' + label+'.jpg')
    # plt.show()
map('Loss_new', TrainLoss, ValLoss, 3)
map('Acc_new', TrainAcc, ValAcc, 1)