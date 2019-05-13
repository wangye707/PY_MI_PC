#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 二维数组查找.py
# @Author: WangYe
# @Date  : 2019/5/13
# @Software: PyCharm
def Find(target, array):
    # write code here
    list =[]
    for temp in array:
        # false = 'false'
        if target in temp:
            list.append(1)
        else:
            list.append(0)
    if 1 in list:
        print("true")
    else:
        print("false")
Find(target=7,array=[[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]])