#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 最大乘积.py
# @Author: WangYe
# @Date  : 2019/2/21
# @Software: PyCharm

# 给定一个无序数组，包含正数、负数和0，要求从中找出3个数的乘积，使得乘积最大，要求时间复杂度：O(n)，空间复杂度：O(1)
a = input()
list_temp = list(a.split(' '))
while ' ' in list_temp:
    list_temp.remove(' ')


print(list_temp)
length = len(list_temp)
###
for u in range(length):
    list_temp[u] = int(list_temp[u])
####
list_temp.sort()
print(list_temp)
res = 0
###
# for i in range(length):
#     for k in range(length):
#         if i == k:
#             continue
#         for m in range(length):
#             if i==m or m ==k:
#                 continue
#             else:
#                 temp = list_temp[i]*list_temp[k]*list_temp[m]
#             if temp>res:
#                 print(i,k,m)
#                 res = temp
#             else:
#                 continue
#case 最大的相乘
res1 = list_temp[length-1]*list_temp[length-2]*list_temp[length-3]
#双负数相乘再加一个最大成正数
res2 = list_temp[0] *list_temp[1] *list_temp[length-1]
if res1>res2:
    res = res1
else:
    res = res2

print(res)