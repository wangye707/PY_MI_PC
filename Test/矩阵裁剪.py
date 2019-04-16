#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 矩阵裁剪.py
# @Author: WangYe
# @Date  : 2019/3/8
# @Software: PyCharm
import numpy as np


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


print(a1.shape)
print(a1[2:2])
print(a1[2:2].shape)