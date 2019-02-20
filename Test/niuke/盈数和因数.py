#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 盈数和因数.py
# @Author: WangYe
# @Date  : 2019/2/20
# @Software: PyCharm
start = ['E:']
wanshu = []
mid = ['G:']
error = ['2']  #我认为不应该包含2，但是答案有2.不包含本身的话，2的因子只有1啊
yingshu = []

for i in range(1,61):
    temp_list = []
    for k in range(1,i):

        if i%k == 0:
            temp_list.append(k)
        else:
            continue
    res = 0
    for j  in range(len(temp_list)):
        res = res + temp_list[j]
    if res == i:
        wanshu.append(i)
    if res >i:
        yingshu.append(i)

last =start+wanshu+mid+error+yingshu

for num in last:
    print(num,end=' ')


