#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 六一儿童节.py
# @Author: WangYe
# @Date  : 2019/3/6
# @Software: PyCharm
"""
输入
3
 2 2 3
 2
 3 1
输出
复制
1
"""

"""
思路：先逆序，然后临近匹配。例如： 4 2 2 1  和  3  2
那最后肯定是   4对3，然后2对2 

"""

N = int(input())
N_list = input()
N_list1 = N_list.split(" ")
child= []
for a in N_list1:
    child.append(int(a))

M = int(input())
M_list = input()
M_list1 = M_list.split(" ")
sugar = []

for b in M_list1:
    sugar.append(int(b))

child.sort(reverse = True)   #逆序
sugar.sort(reverse = True)

X = []  #存储巧克力
Y = []   #存储孩子坐标
temp = 0
for q in range(M):
    for w in range(temp,N):

        if  sugar[q] >= child[w]:
            X.append(q)

            temp = w   #防止supar中出现重复匹配
            break
    if temp == N-1:  #当直接到最后一个child时，直接跳出
        break


if len(X)>N:  #防止sugar的数量大于child
    print(N)
else:
    print(len(X))  #正常输出
