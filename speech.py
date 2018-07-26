
# coding: utf-8

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import random

# Model Hyperparameters
# 第一层输入，MFCC信号
tf.flags.DEFINE_integer("n_inputs",  40,  "Number of MFCCs (default: 40)")
# cell个数
tf.flags.DEFINE_integer("n_hidden",  300,  "Number of cells (default: 300)")
# 分类数
tf.flags.DEFINE_integer("n_classes",  23,  "Number of classes (default: 22)")
# 学习率
tf.flags.DEFINE_float("lr",  0.1,  "Learning rate (default: 0.05)")
# dropout参数
tf.flags.DEFINE_float("dropout_keep_prob",  0.5,  "Dropout keep probability (default: 0.5)")

# Training parameters
# 批次大小
tf.flags.DEFINE_integer("batch_size",  10,  "Batch Size (default: 50)")
# 迭代周期
tf.flags.DEFINE_integer("num_epochs",  100,  "Number of training epochs (default: 100)")
# 多少step测试一次
tf.flags.DEFINE_integer("evaluate_every",  10,  "Evaluate model on dev set after this many steps (default: 10)")
# 多少step保存一次模型
tf.flags.DEFINE_integer("checkpoint_every",  500,  "Save model after this many steps (default: 500)")
# 最多保存多少个模型
tf.flags.DEFINE_integer("num_checkpoints",  2,  "Number of checkpoints to store (default: 2)")

# flags解析
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

# 加载训练用的特征和标签
# train_features = np.load('train_features.npy')
# train_labels = np.load('train_labels.npy')
train_features = np.load('half_features.npy')
train_labels = np.load('half_labels.npy')
#参数：批次，序列号（分帧的数量），每个序列的数据(batch, step, input)
#(50, 199, 40)

# 计算最长的step,分为step帧
wav_max_len = max([len(feature) for feature in train_features])
print("max_len:", wav_max_len)

# 填充0
tr_data = []
for mfccs in train_features:
    while len(mfccs) < wav_max_len:
        mfccs.append([0] * FLAGS.n_inputs)
    tr_data.append(mfccs)
tr_data = np.array(tr_data)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(tr_data)))
x_shuffled = tr_data[shuffle_indices]
y_shuffled = train_labels[shuffle_indices]

# 数据集切分为两部分
dev_sample_index = -1 * int(0.2 * float(len(y_shuffled)))
train_x,  test_x = x_shuffled[:dev_sample_index],  x_shuffled[dev_sample_index:]
train_y,  test_y = y_shuffled[:dev_sample_index],  y_shuffled[dev_sample_index:]

x = tf.placeholder("float",  [None,  wav_max_len,  FLAGS.n_inputs])
y = tf.placeholder("float",  [None])
dropout = tf.placeholder(tf.float32)

# 定义RNN网络
# 初始化权值和偏置n_hidden300
weights = tf.Variable(tf.truncated_normal([FLAGS.n_hidden,  FLAGS.n_classes],  stddev=0.1))
biases = tf.Variable(tf.constant(0.1,  shape=[FLAGS.n_classes]))

# 网络层数
# num_layers = 3
# num_layers = 1
# def grucell():
#     cell = tf.contrib.rnn.GRUCell(FLAGS.n_hidden)
# #     cell = tf.contrib.rnn.LSTMCell(FLAGS.n_hidden)
# #     cell = tf.contrib.rnn.DropoutWrapper(cell,  output_keep_prob=dropout)
#     return cell
# cell = tf.contrib.rnn.MultiRNNCell([grucell() for _ in range(num_layers)])
cell = tf.contrib.rnn.LSTMCell(FLAGS.n_hidden)
outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

# 预测值
prediction = tf.nn.softmax(tf.matmul(final_state[0], weights) + biases)

# labels转one_hot格式
one_hot_labels = tf.one_hot(indices=tf.cast(y,  tf.int32),  depth=FLAGS.n_classes)

# loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=one_hot_labels))

# optimizer
lr = tf.Variable(FLAGS.lr,  dtype=tf.float32,  trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1),  tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,  tf.float32))


def batch_iter(data,  batch_size,  num_epochs,  shuffle=True):
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:", num_batches_per_epoch)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size,  data_size)
            yield shuffled_data[start_index:end_index]


# Initializing the variables
init = tf.global_variables_initializer()
# 定义saver
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # Generate batches
    batches = batch_iter(list(zip(train_x,  train_y)),  FLAGS.batch_size,  FLAGS.num_epochs)

    for i, batch in enumerate(batches):
        i = i + 1
        x_batch,  y_batch = zip(*batch)
        # sess.run([optimizer],  feed_dict={x: x_batch,  y: y_batch,  dropout: FLAGS.dropout_keep_prob})
        sess.run([optimizer], feed_dict={x: x_batch, y: y_batch})
        # 测试
        # if i % FLAGS.  == 0:
        if i % 50 == 0:
            sess.run(tf.assign(lr,  FLAGS.lr * (0.90 ** (i // FLAGS.evaluate_every))))
            learning_rate = sess.run(lr)
            tr_acc, _loss = sess.run([accuracy, cross_entropy], feed_dict={x: train_x, y: train_y})
            ts_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
            # tr_acc,  _loss = sess.run([accuracy,  cross_entropy],  feed_dict={x: train_x,  y: train_y,  dropout: 1.0})
            # ts_acc = sess.run(accuracy,  feed_dict={x: test_x,  y: test_y,  dropout: 1.0})
            print("Iter {}, loss {:.5f}, tr_acc {:.5f}, ts_acc {:.5f}, lr {:.5f}".format(i, _loss, tr_acc, ts_acc, learning_rate))

        # 保存模型
        if i % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, "sounds_models/model", global_step=i)
            print("Saved model checkpoint to {}\n".format(path))