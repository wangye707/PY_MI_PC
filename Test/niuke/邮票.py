#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 邮票.py
# @Author: WangYe
# @Date  : 2019/2/19
# @Software: PyCharm
a = [0.8,0.8,0.8,0.8,0.8]
b = [1,1,1,1]
c= [1.8,1.8,1.8,1.8,1.8,1.8]
#print(d)
num = 0
res = []
#排列组合函数
list_out = []

for i in range(len(a)+1):
    for j in range(len(b)+1):
        for k in range(len(c)+1):
            res.append(float(8*i)+float(10*j)+float(18*k))

#print(res)
res.sort()
for i in range(len(res)):
    if res[i] in list_out:
        continue
    else:
        list_out.append(res[i])
#print(list_out)
print(len(list_out)-1)
