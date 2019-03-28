#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : CNN_single.py
# @Author: WangYe
# @Date  : 2019/3/11
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
# tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
#                      'Steps to validate and print loss')
#
# For distributed
tf.app.flags.DEFINE_string("ps_hosts","localhost:11111",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:111112,localhost:111113,localhost:111114",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
tf.app.flags.DEFINE_string("cuda", "0", "specify gpu")
#FLAGS = tf.app.flags.FLAGS
if FLAGS.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda





"""
权重初始化
初始化为一个接近0的很小的正数
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

"""
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size）
池化用简单传统的2x2大小的模板做max pooling
"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize(pool大小)        : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides(pool滑动大小)   : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.

start = time.clock() #计算开始时间
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True) #MNIST数据输入
def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            print("test")
            """
            第一层 卷积层
            
            x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
            """
            x = tf.placeholder(tf.float32,[None, 784])
            x_image = tf.reshape(x, [-1, 28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            # x_image -> [batch, in_height, in_width, in_channels]
            #            [batch, 28, 28, 1]
            # W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
            #            [5, 5, 1, 32]
            # output  -> [batch, out_height, out_width, out_channels]
            #            [batch, 28, 28, 32]
            h_pool1 = max_pool_2x2(h_conv1)
            # h_conv1 -> [batch, in_height, in_weight, in_channels]
            #            [batch, 28, 28, 32]
            # output  -> [batch, out_height, out_weight, out_channels]
            #            [batch, 14, 14, 32]

            """
            第二层 卷积层
            
            h_pool1(batch, 14, 14, 32) -> h_pool2(batch, 7, 7, 64)
            """
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            # h_pool1 -> [batch, 14, 14, 32]
            # W_conv2 -> [5, 5, 32, 64]
            # output  -> [batch, 14, 14, 64]
            h_pool2 = max_pool_2x2(h_conv2)
            # h_conv2 -> [batch, 14, 14, 64]
            # output  -> [batch, 7, 7, 64]

            """
            第三层 全连接层
            
            h_pool2(batch, 7, 7, 64) -> h_fc1(1, 1024)
            """
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            """
            Dropout
            
            h_fc1 -> h_fc1_drop, 训练中启用，测试中关闭
            """
            keep_prob = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            """
            第四层 Softmax输出层
            """
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            """
            训练和评估模型
            
            ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
            """
            y_ = tf.placeholder("float", [None, 10])
            loss_value = -tf.reduce_sum(y_ * tf.log(y_conv))  # 计算交叉熵
            optimizer = tf.train.GradientDescentOptimizer(0.10)  # 使用adam优化器来以0.0001的学习率来进行微调
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 判断预测标签和实际标签是否匹配
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # sess = tf.Session()  # 启动创建的模型
            # sess.run(tf.initialize_all_variables())  # 旧版本


            # sess.run(tf.global_variables_initializer()) #初始化变量
            grads_and_vars = optimizer.compute_gradients(loss_value)

            if issync == 1:
                # 同步模式计算更新梯度
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=5,
                                                        #                     replica_id=FLAGS.task_index,
                                                        total_num_replicas=5,
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars,
                                                  global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                # 异步模式计算更新梯度
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step)

            init_op = tf.initialize_all_variables()

            # saver = tf.train.Saver()
            tf.summary.scalar('cost', loss_value)
            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 #  logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=None,
                                 #  saver=saver,
                                 global_step=global_step,
                                 # save_model_secs=60
                                 )

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            step = 0

            # with tf.Session() as sess:
            #     tf.global_variables_initializer().run()

            while step < 1000:  # 开始训练模型，循环训练5000次
                batch = mnist.train.next_batch(120)  # batch大小设置为50
                _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                           feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                # _loss, __ = sess.run([train_op, grads_and_vars],
                #                      feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

                if step % 100 == 0:
                    acc, loss = sess.run([accuracy, loss_value],
                                         feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    # train_accuracy = accuracy.eval(session = sess,
                    #                                feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
                    print("step %d, train_accuracy %g" % (step, acc))
                # train_step.run(session = sess, feed_dict = {x:batch[0], y_:batch[1],
                #                keep_prob:0.5}) #神经元输出保持不变的概率 keep_prob 为0.5

                # print("test accuracy %g" %accuracy.eval(session = sess,
                #       feed_dict = {x:mnist.test.images, y_:mnist.test.labels,
                #                    keep_prob:1.0})) #神经元输出保持不变的概率 keep_prob 为 1，即不变，一直保持输出

            end = time.clock()  # 计算程序结束时间
            out = (end - start)
            print("running time is", out, "s")


if __name__ == "__main__":
    tf.app.run()