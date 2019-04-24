#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 矩阵裁剪.py
# @Author: WangYe
# @Date  : 2019/3/8
# @Software: PyCharm
import numpy as np
import tensorflow as tf

a = [[[[1,2,3],
     [1,2,3],
     [1,2,3]],[[1,2,3],
     [1,2,3],
     [1,2,3]]],[[[1,2,3],
     [1,2,3],
     [1,2,3]],[[1,2,3],
     [1,2,3],
     [1,2,3]]]]
a1 = np.array(a)
#a1 = a
b = [[1,2,3],
     [1,2,3]]
c = [[1,2],
     [1,2],[1,2]]

d = [[[1,1],
     [1,1],[1,1]],
     [[2, 2],
      [2, 2], [2, 2],],[[1,2],
     [1,2],[1,2]]
     ]

dx = np.asarray(d)
# dx = dx.reshape(3,2,1)
# print(dx)
# print(dx.shape)
# dq = np.asarray([dx[0]])
# print(dq.shape)
print(dx[2])
# cx =np.ravel(c)
# cw = np.reshape(a,(3,1,2))
# print(cx)

# d = tf.constant([[1,2,3],
#      [1,2,3]],tf.float64)
# e = tf.constant([[1,2,3],
#      [1,2,3]],tf.float64)
#
# print(d)
# print(e)
# with tf.Session() as sess:
#      # dis1 = sess.run(tf.square(e-d))
#      # dis2 = sess.run(tf.reduce_sum(dis1))#
#      dis = sess.run(tf.sqrt(tf.reduce_sum(tf.square(d-e), 2)))
#      # dis3 = sess.run(tf.sqrt(dis2))#
#      #print(dis1,"++++++",dis2,"+++++",dis3)
#      print(dis)



# b1 = np.array(b)
# c1 = np.array(c)
# print(c1.shape)
# c2 = np.pad(c1,pad_width=((0,0),(1,0)),mode="constant",constant_values=(0,0))
# #                       行填充    列填充
#
# # c2 = np.pad(c1,pad_width=((0,0),(1,0)),mode="reflect",constant_values=(0,0))
# print(c2)
# print(c2.shape)
# print(type(b1))#<class 'numpy.ndarray'>
# print(b1.shape)#(2, 3)
# x1 = b1.shape
# x2 = c1.shape
# # print(type(x1))#<class 'tuple'>
# temp = []
# temp.append(x1[0])
# temp.append(x2[1])
# print(max(temp))




# print(a1.shape)
# print(a1[2:2])
# print(a1[2:2].shape)