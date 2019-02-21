#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 反序数.py
# @Author: WangYe
# @Date  : 2019/2/20
# @Software: PyCharm

back_num_str = []
num_str = []
for i in range(1000,10000):
    back_num = i * 9
    res = 0
    back_num_str = str(back_num)

    num_str = str (i)
    length = len(num_str)


    for k in range(length):

        if back_num_str[k] == num_str[length - k-1]:
            res = res + 1
        else:
            break
    if res == length:
        print(i)


