#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : homework3.py
# @Author: WangYe
# @Date  : 2018/4/22
# @Software: PyCharm
# 微博文字的性别识别
import jieba
import os
import pickle  # 持久化
from numpy import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法


def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content


def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)


def segText(inputPath, resultPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            #  print(eachFile)
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            # content = str(content)
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()

            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件


def bunchSave(inputFile, outputFile):
    catelist = os.listdir(inputFile)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签
            bunch.filenames.append(fullName)  # 保存当前文件的路径
            bunch.contents.append(readFile(fullName).strip())  # 保存文件词向量
    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)
        # pickle.dump(obj, file, [,protocol])函数的功能：将obj对象序列化存入已经打开的file中。
        # obj：想要序列化的obj对象。
        # file:文件名称。
        # protocol：序列化使用的协议。如果该项省略，则默认为0。如果为负值或HIGHEST_PROTOCOL，则使用最高的协议版本


def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList


def getTFIDFMat(inputPath, stopWordList, outputPath):  # 求得TF-IDF向量
    bunch = readBunch(inputPath)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    # print(tfidfspace)
    '''读取tfidfspace'''
    tfidfspace_out = str(tfidfspace)
    saveFile("D:/code/python/work/Text_data/tfidfspace.txt",tfidfspace_out)
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace_arr = str(vectorizer.fit_transform(bunch.contents))
    saveFile("D:/code/python/work/Text_data/tfidfspace_arr.txt", tfidfspace_arr)
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
    tfidfspace_vocabulary = str(vectorizer.vocabulary_)
    saveFile("D:/code/python/work/Text_data/tfidfspace_vocabulary.txt", tfidfspace_vocabulary)
    '''over'''
    writeBunch(outputPath, tfidfspace)


def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath):
    bunch = readBunch(testSetPath)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                      vocabulary={})
    '''
    读取testSpace
    '''
    testSpace_out = str(testSpace)
    saveFile("D:/code/python/work/Text_data/testSpace.txt",testSpace_out)
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    testSpace_arr = str(testSpace.tdm)
    trainbunch_vocabulary = str(trainbunch.vocabulary)
    saveFile("D:/code/python/work/Text_data/testSpace_arr.txt", testSpace_arr)
    saveFile("D:/code/python/work/Text_data/trainbunch_vocabulary.txt", trainbunch_vocabulary)
    # 持久化
    writeBunch(testSpacePath, testSpace)


def bayesAlgorithm(trainPath, testPath):
    trainSet = readBunch(trainPath)
    # print(trainSet)
    testSet = readBunch(testPath)
    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
    # alpha:0.001 alpha 越小，迭代次数越多，精度越高
    # print(shape(trainSet.tdm))  #输出单词矩阵的类型
    '''处理bat文件'''
    tfidfspace_out_arr=str(trainSet.tdm)  #处理
    tfidfspace_out_word = str(trainSet)
    saveFile("D:/code/python/work/Text_data/tfidfspace_out_arr.txt",tfidfspace_out_arr)   #矩阵形式的train_set.txt
    saveFile("D:/code/python/work/Text_data/tfidfspace_out_word.txt",tfidfspace_out_word)  #文本形式的train_set.txt

    testspace_out_arr = str(testSet)
    testspace_out_word = str(testSet.label)
    saveFile("D:/code/python/work/Text_data/testspace_out_arr.txt",testspace_out_arr)
    saveFile("D:/code/python/work/Text_data/testspace_out_word.txt",testspace_out_word)

    '''处理结束'''
    # print(testSet.tdm)
    predicted = clf.predict(testSet.tdm)
    # print(predicted)
    total = len(predicted)
    rate = 0
    for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
        if flabel != expct_cate:
            rate += 1
            print(fileName, ":实际类别：", flabel, "-->预测类别：", expct_cate)
    print("erroe rate:", float(rate) * 100 / float(total), "%")






# 分词，第一个是分词输入，第二个参数是结果保存的路径
segText("D:/code/python/work/Text_data/data/", "D:/code/python/work/Text_data/segResult/")
bunchSave("D:/code/python/work/Text_data/segResult/", "D:/code/python/work/Text_data/train_set.dat")  # 输入分词，输出分词向量
stopWordList = getStopWord("D:/code/python/work/Text_data/stop/stopword.txt")  # 获取停用词
getTFIDFMat("D:/code/python/work/Text_data/train_set.dat", stopWordList, "D:/code/python/work/Text_data/tfidfspace.dat")  # 输入词向量，输出特征空间

# 训练集
segText("D:/code/python/work/Text_data/test1/", "D:/code/python/work/Text_data/test_segResult/")  # 分词
bunchSave("D:/code/python/work/Text_data/test_segResult/", "D:/code/python/work/Text_data/test_set.dat")
getTestSpace("D:/code/python/work/Text_data/test_set.dat", "D:/code/python/work/Text_data/tfidfspace.dat", stopWordList,
             "D:/code/python/work/Text_data/testspace.dat")
bayesAlgorithm("D:/code/python/work/Text_data/tfidfspace.dat", "D:/code/python/work/Text_data/testspace.dat")