#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 111.py
# @Author: WangYe
# @Date  : 2019/5/11
# @Software: PyCharm
import numpy as np
def wider(seq, l=5, n=5):
    """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
    key = "HRKDE"
    tensor = []
    for i in range(l):
        d2 = []
        for j in range(l):
            d1 = [1 if (j < len(seq) and i < len(seq) and key[x] == seq[i] and key[x] == seq[j]) else 0 for x in
                  range(n)]

            d2.append(d1)
        tensor.append(d2)
    print(tensor)
    print(len(tensor[0]))
    print(len(tensor[0][0]))
    # print(len(tensor[0][0][0]))

    # print(tensor[0].shape())
    return np.array(tensor)
SS = "HRHRH"
# wider(SS)
A = [1,2,3,4]
print(A[-1])