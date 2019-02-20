#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : multiple_regression.py
# @Author: WangYe
# @Date  : 2019/2/20
# @Software: PyCharm

print("韩龙飞是傻逼")

from HanLong.xlsx_prepare import input_data_prepare
from sklearn import linear_model

X = input_data_prepare()[0]
Y = input_data_prepare()[1]
print("X",X)
print("Y",Y)
def regression_for_hanlong():
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print("coefficients:", regr.coef_)  # b1,...,bp（与x相结合的各个参数）
    print("intercept:", regr.intercept_)  # b0（截面）

    x_pred = [[137.124, 0.1,59.493,13.196,2.649,66.935]]
    y_pred = regr.predict(x_pred)  # 预测
    print("y_pred:", y_pred)

regression_for_hanlong()