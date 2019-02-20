#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:09:43 2017
this file is used to store the batch_input param
@author: root

"""
batch_size=50
start=0
end=start+batch_size
#have read length
now_size=0
text_count=0
#当前相关联的qid
now_relate_qid=0
#当前不相关联的qid
now_not_relate_qid=35
#query长度限制
query_word_length=5
#document长度限制
doc_word_length=200
#没有关联的qid的开始位置
start_not_relate_doc=now_not_relate_qid
#向量长度
word_vec_length=50
