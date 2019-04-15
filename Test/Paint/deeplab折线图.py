#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : RNN折线图.py
# @Author: WangYe
# @Date  : 2019/3/26
# @Software: PyCharm

import matplotlib.pyplot as plt
y1=[418000,357900,279300,301580,510350,604810]
x1=[4,6,8,10,15,20]
x2=[4,6,8,10,15,20]
x3=[4,6,8,10,15,20]
y2=[418000,418000,418000,418000,418000,418000]
y3=[527100,527100,527100,527100,527100,527100]
plt.plot(x1,y1,label='new SSGD',linewidth=3,color='r',marker='o',
markerfacecolor='blue',markersize=12)
plt.plot(x2,y2,label='SSGD')
plt.plot(x3,y3,label='Single')
plt.xlabel('m')
plt.ylabel('costing time')
plt.title('new SSGD and SSGD costing time(encoder-DeepLabv3+)')
plt.legend()
plt.show()
