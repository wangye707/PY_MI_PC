#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : minist.py
# @Author: WangYe
# @Date  : 2019/1/11
# @Software: PyCharm
from __future__ import print_function
import os
from PIL import Image

#from paddle.v2.plot import Ploter
import numpy
import paddle
import paddle.fluid as fluid



def softmax_regression():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(
        input=img, size=10, act='softmax')
    return predict

def multilayer_perceptron():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction

def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction

def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # predict = softmax_regression() # uncomment for Softmax回归
    # predict = multilayer_perceptron() # uncomment for 多层感知器
    predict = convolutional_neural_network() # uncomment for LeNet5卷积神经网络
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return predict, [avg_cost, acc]

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

BATCH_SIZE = 64

train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

def event_handler(pass_id, batch_id, cost):
    print("Pass %d, Batch %d, Cost %f" % (pass_id,batch_id, cost))



# train_prompt = "Train cost"
# test_prompt = "Test cost"
# cost_ploter = Ploter(train_prompt, test_prompt)
#
# # event_handler to plot a figure
# def event_handler_plot(ploter_title, step, cost):
#     cost_ploter.append(ploter_title, step, cost)
#     cost_ploter.plot()


use_cuda = True # set to True if training with GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

prediction, [avg_loss, acc] = train_program()

img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_loss)

PASS_NUM = 5
epochs = [epoch_id for epoch_id in range(PASS_NUM)]

save_dirname = "recognize_digits.inference.model"

def train_test(train_test_program,
                   train_test_feed, train_test_reader):
    acc_set = []
    avg_loss_set = []
    for test_data in train_test_reader():
        acc_np, avg_loss_np = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=[acc, avg_loss])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    # get test acc and loss
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

main_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)

lists = []
step = 0
for epoch_id in epochs:
    for step_id, data in enumerate(train_reader()):
        metrics = exe.run(main_program,
                          feed=feeder.feed(data),
                          fetch_list=[avg_loss, acc])
        if step % 100 == 0:
            print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[0]))
            #event_handler_plot(train_prompt, step, metrics[0])
        step += 1

    # test for epoch
    avg_loss_val, acc_val = train_test(train_test_program=test_program,
                                       train_test_reader=test_reader,
                                       train_test_feed=feeder)

    print("Test with Epoch %d, avg_cost: %s, acc: %s" %(epoch_id, avg_loss_val, acc_val))
    #event_handler_plot(test_prompt, step, metrics[0])

    lists.append((epoch_id, avg_loss_val, acc_val))
    if save_dirname is not None:
        fluid.io.save_inference_model(save_dirname,
                                      ["img"], [prediction], exe,
                                      model_filename=None,
                                      params_filename=None)

        # find the best pass
best = sorted(lists, key=lambda list: float(list[1]))[0]
print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

