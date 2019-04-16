#!D:/workplace/python

# -*- coding: utf-8 -*-

# @File  : handwriting_STFTCNN.py

# @Author: Li Qingpei

# @Date  : 2018/11/26

# @Software: PyCharm
import os

import numpy as np

import tensorflow as tf

import keras

import tensorflow.contrib.rnn as rnn

import readData

# import tensorflow as tf
#
# import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' #use GPU with ID=0

config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 1 # maximun alloc gpu50% of MEM

# config.gpu_options.allow_growth = True #allocate dynamically

# sess = tf.Session(config = config)
''''
代码改动1
'''
TIME_STEPS=10
BATCH_SIZE=50
HIDDEN_UNITS1=30
HIDDEN_UNITS=3
LEARNING_RATE=0.001
EPOCH=50

TRAIN_EXAMPLES=468
TEST_EXAMPLES=312
train_X, test_X, train_y, test_y, lable = readData.readDatatrain1()

#======使用两个数据集，一个是训练集，一个是测试集
#print(train_X.shape, test_X.shape, train_y.shape, test_y.shape, lable.shape)
#  (468, 91, 875, 1) (312, 91, 875, 1) (468, 1) (312, 1) (780, 1)
#print('data has readed!!!')
N = 26  # N为类别个数
acc=np.zeros((9,1))
re=[]
ma=[]
# 将一维label转换为二维label
train_y = keras.utils.to_categorical((train_y - 1), num_classes=N)
test_y = keras.utils.to_categorical((test_y - 1), num_classes=N)
train_x = train_X/255
test_x = train_X/255
train_x = train_x[:,:,:,-1]

test_x = test_x[:,:,:,-1]
#train_y = train_y[:,-1]
#test_y = test_y[:,-1]


print('0000',train_x.shape, test_x.shape, train_y.shape, test_y.shape)
#开始给你写LSTM,你就是猪猪猪猪猪猪猪猪

graph=tf.Graph()
with graph.as_default():

    #------------------------------------construct LSTM------------------------------------------#
    #place hoder
    X_p=tf.placeholder(dtype=tf.float32,shape=(BATCH_SIZE,91,875),name="input_placeholder")

    y_p=tf.placeholder(dtype=tf.float32,shape=(BATCH_SIZE,26),name="pred_placeholder")

    #lstm instance
    lstm_cell=rnn.BasicLSTMCell(num_units=50)
    lstm_cell1=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_cell3 = rnn.BasicLSTMCell(num_units=26)


    multi_lstm=rnn.MultiRNNCell(cells=[lstm_cell,lstm_cell1,lstm_cell3])
    #print(multi_lstm.shape)
    #initialize to zero
    init_state=multi_lstm.zero_state(batch_size=BATCH_SIZE,dtype=tf.float32)

    #dynamic rnn
    outputs,states=tf.nn.dynamic_rnn(cell=multi_lstm,inputs=X_p,initial_state=init_state,dtype=tf.float32)
    print(outputs.shape) #(128, 91, 26)
    #print(states)
    h=outputs[:,-1,:]
    print(h.shape)#(128, 26)
    print(y_p.shape)#(128, 26)

    #cross_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_p, logits=h)
    correct_prediction = tf.equal(tf.argmax(h,1), tf.argmax(y_p,1))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction, "float"))


    mse=tf.losses.mean_squared_error(labels=y_p,predictions=h)
    #print(loss.shape)
    optimizer=tf.train.AdamOptimizer(0.001).minimize(loss=mse)


    init=tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1,EPOCH+1):
        results = np.zeros(shape=(TEST_EXAMPLES, 1))
        train_losses=[]
        test_losses=[]

        print("epoch:",epoch)
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            # print('111',train_x[j*BATCH_SIZE:(j+1)*BATCH_SIZE].shape)
            # print('222', train_y[j * BATCH_SIZE:(j + 1) * BATCH_SIZE].shape)
            _,train_loss=sess.run(
                    fetches=(optimizer,mse),
                    feed_dict={
                            X_p:train_x[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:train_y[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            train_losses.append(train_loss)
        print("average training loss:", sum(train_losses) / len(train_losses))
        print(acc)

        for j in range(TEST_EXAMPLES//BATCH_SIZE):
            results,test_loss,acc=sess.run(
                    fetches=(h,mse,accuracy),
                    feed_dict={
                            X_p:test_x[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:test_y[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            #results[j*BATCH_SIZE:(j+1)*BATCH_SIZE]=result
            test_losses.append(test_loss)
        print("average test loss:", sum(test_losses) / len(test_losses))
        print(accuracy)

