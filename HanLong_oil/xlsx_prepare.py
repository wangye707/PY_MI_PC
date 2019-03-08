#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : xlsx_prepare.py
# @Author: WangYe
# @Date  : 2019/2/20
# @Software: PyCharm
import numpy
import xlrd

path = r'D:\code\python\HanLong\NEW_DATA.xlsx'


def open_xlrd(path):  #表格读取
    try:
        data =xlrd.open_workbook(path)
        return data
    except:
        print("文件读取失败")
#
# def do_xlrd():     #表格处理
#
#     data = open_xlrd(path)
#     table = data.sheets()[0]
#     #GR= table.row_values()
#     # POR = table.row_values(1)
#     # SH = table.row_values(2)
#     # CNL = table.row_values(3)
#     # DEN = table.row_values(4)
#     # ACFT = table.row_values(5)
#     # DTSM =table.row_values(6)
#
#     # return GR,POR,SH,CNL,DEN,ACFT,DTSM
#     return GR
def do_xlrd():     #表格处理

    res_list =[]
    data = open_xlrd(path)

    table = data.sheets()[0]
    deep = table.col_values(0)
    #print(len(deep))
    for i in range(1,len(deep)):
        res_list.append(table.row_values(i))
    return res_list



def input_data_prepare():   #将数据处理成为机器学习输入类型
    data = do_xlrd()
    #print(data)
    res = numpy.mat(data)
    X = res[:, :-1]  #取出所有自变量
    Y = res[:, -1]  # 取出所有因变量

    for i in Y:
        print(i)


    #print(Y)
    # print("x",X)
    # print("Y",Y)
    return X,Y
input_data_prepare()