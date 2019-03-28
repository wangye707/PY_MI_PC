#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : CNN折线图.py
# @Author: WangYe
# @Date  : 2019/3/7
# @Software: PyCharm

import matplotlib.pyplot as plt
y1=[1194,996,504,898,1300,1900]
x1=[4,6,8,10,15,20]
x3=[4,6,8,10,15,20]#单机代码
x2=[4,6,8,10,15,20]
y2=[1194,1194,1194,1194,1194,1194]

y3=[1799,1799,1799,1799,1799,1799]
plt.plot(x1,y1,label='new SSGD',linewidth=3,color='r',marker='o',
markerfacecolor='blue',markersize=12)
plt.plot(x2,y2,label='SSGD')
plt.plot(x3,y3,label='Single')
plt.xlabel('m')
plt.ylabel('costing time')
plt.title('new SSGD and SSGD costing time(CNN)')
plt.legend()
plt.show()


