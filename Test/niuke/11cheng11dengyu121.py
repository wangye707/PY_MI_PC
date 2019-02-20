#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 11cheng11dengyu121.py
# @Author: WangYe
# @Date  : 2019/2/19
# @Software: PyCharm
# 题目描述
# 打印所有不超过n（n<256）的，其平方具有对称性质的数。如11*11=121。
# 输入描述:
# 无
# 输出描述:
# 每行一个数，表示对称平方数。
n = 256
#print(10**2)
top = 0
for i in range(1,256):
    mul = i * i

    if mul > 10: #在两位数开始找
        for k in range(1,5):     #找出乘数的最高位
            temp = mul / (10**k)
            if 1 <= temp <= 10:
                top = int(temp)
                break
            else:
                continue
    #print(top)
    if top == (mul % 10):
        print(i,mul)