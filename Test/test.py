#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: WangYe
# @Date  : 2019/4/23
# @Software: PyCharm
import jieba
def get_most_similar_list(inputQuestion, candQuestionDicts, stopwords_file_path):
    '''
    inputQuestion为用户输入问题，字符串类型
    candQuestions待选问题的一个列表，每一个元素为一个字符串类型
    stopwords_file_path为停用词文件的路径
    '''

    # 加载停用词
    stopwords = [line.strip() for line in open(stopwords_file_path, 'r', encoding='utf-8').readlines()]

    # [{"sentence": "fdsfdsafdsa", "zid": "abc123"},  {},  {}]
    candQuestions = [candQuestionDict['sentence'] for candQuestionDict in candQuestionDicts]
    zids = [candQuestionDict['zid'] for candQuestionDict in candQuestionDicts]
    candQuestions = [jieba.lcut(candQuestion) for candQuestion in candQuestions]
    # 构建词向量
    # # dictionary = corpora.Dictionary(candQuestions)
    # corpus = [dictionary.doc2bow(question) for question in candQuestions]
    # tfidf = models.TfidfModel(corpus, id2word=dictionary)
    # index = similarities.MatrixSimilarity(tfidf[corpus])

    # 相似度匹配
    inputQuestion = jieba.lcut(inputQuestion.strip())
    inputQuestion = [t for t in inputQuestion if t not in stopwords]
    # inputQuestion = dictionary.doc2bow(inputQuestion)
    # te = tfidf[inputQuestion]
    # sim = index[te]
    # indices = np.argsort(-sim, kind='heapsort')[:5]
    # results = [zids[ind] for ind in indices]
    return inputQuestion

import time
sentence = "WY LOVE LQP"
for char in sentence.split():
   allChar = []
   for y in range(12, -12, -1):
       lst = []
       lst_con = ''
       for x in range(-30, 30):
            formula = ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3
            if formula <= 0:
                lst_con += char[(x) % len(char)]
            else:
                lst_con += ' '
       lst.append(lst_con)
       allChar += lst
   print('\n'.join(allChar))
   time.sleep(1)
