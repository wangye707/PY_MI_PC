#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 算数字.py
# @Author: WangYe
# @Date  : 2019/2/20
# @Software: PyCharm
#abc、bcc 是两个三位数，且有：abc+bcc=532。
for a in range(10):
    for b in range(10):
        for c in  range(10):
            abc = (100*a)+(10*b)+c
            bcc = (100*b)+(10*c)+c
            if (abc+bcc)==532:
                print(a,b,c)