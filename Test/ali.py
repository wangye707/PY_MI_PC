#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : ali.py
# @Author: WangYe
# @Date  : 2019/4/2
# @Software: PyCharm
'''
/*
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

示例:

输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
*/

'''
input_list = input()
list1 = input_list.strip("[").strip("]").split(",")

new_list = []

for k in list1:
    new_list.append(int(k))    #将输入的str的list转换成真正的list


step = 0
max = len(new_list)  #

final_list = [0]

#now是指从哪一步开始，题目要求就是从0开始
#[1,2,1,4,2,3]
def find_keys(now):
    out = now            #用于递归查找
    k =  new_list[now] #now = 0  k =2

    temp = 0
    for m in range(now+1,k+now+1):    #查找下一个最长的路径
        if new_list[m] >= temp:
            temp = new_list[m]
            out = m
    #print(out)
    final_list.append(out)
    if (now + temp) >=  (max - 2):
        return final_list        #满足条件之后跳出
    else:
        return find_keys(now = out)       #不满足继续递归

out = find_keys(0)
print(out)
print(len(out))