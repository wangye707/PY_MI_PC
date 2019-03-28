#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : RNN折线图.py
# @Author: WangYe
# @Date  : 2019/3/26
# @Software: PyCharm

import matplotlib.pyplot as plt
y1=[664,422,399,468,690,702]
x1=[4,6,8,10,15,20]
x2=[4,6,8,10,15,20]
x3=[4,6,8,10,15,20]
y2=[660,660,660,660,660,660]
y3=[723,723,723,723,723,723]
plt.plot(x1,y1,label='new SSGD',linewidth=3,color='r',marker='o',
markerfacecolor='blue',markersize=12)
plt.plot(x2,y2,label='SSGD')
plt.plot(x3,y3,label='Single')
plt.xlabel('m')
plt.ylabel('costing time')
plt.title('new SSGD and SSGD costing time(RNN)')
plt.legend()
plt.show()
