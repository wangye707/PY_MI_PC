#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 空格替换.py
# @Author: WangYe
# @Date  : 2019/5/13
# @Software: PyCharm
def replaceSpace( s):
    # write code here
    str(s)
    list = []
    for l in s:
        if l != ' ':
            list.append(l)
        else:
            list.append('%20')
    m = ''
    for k in list:
        m = m+k
    print(m)
replaceSpace('hello world')