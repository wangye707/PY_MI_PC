#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:04:29 2017
read files new
@author: root
"""
import os
import numpy as np
import tensorflow as tf
import re
import linecache
import store_batch_parm
#import xrange
data_dir='./title'
docid_doc='docid_doc.txt'
embed_wiki_pdc_d50='embed_wiki_pdc_d50'
embed_wiki_pdc_d50_norm='embed_wiki_pdc_d50_norm'
qid_query='qid_query.txt'
batch_size=store_batch_parm.batch_size
doc_word_length=store_batch_parm.doc_word_length
query_word_length=store_batch_parm.query_word_length
word_vec_length=store_batch_parm.word_vec_length
word_dict='word_dict.txt'

def read_word_by_did_old(data_dir,docid_doc):
    file_name=os.path.join(data_dir,docid_doc)
    with open(file_name,'r') as f:
        lines=f.readlines()
        qid_words=dict()
        for line in lines:
            word_split=line.split()
            temp_word_id=[]
#            for word_split,i in word_split:
            for i,word_id in enumerate(word_split):
                if i>0:
                    temp_word_id.append(int(word_id))
#            change the type of  temp_word_id to float64
            temp_word_id_int=temp_word_id
            qid_words[word_split[0]]=temp_word_id_int
    return qid_words
read_word_by_did_old(data_dir,docid_doc)	
#==============================================================================
# test=['FBIS3-10001','FAIS3-100']
# my_find=dict()
# for find in test:
#     key,value=doc_words=read_word_by_did(data_dir,find,2)
#     my_find[key]=value
# 
# print('my_find:',my_find)
#==============================================================================

def read_word_by_qid(data_dir,qid_query):
    file_name=os.path.join(data_dir,qid_query)
    with open(file_name,'r') as f:
        lines=f.readlines()
        qid_words=dict()
        for line in lines:
            word_split=line.split()
            temp_word_id=[]
#            for word_split,i in word_split:
            for i,word_id in enumerate(word_split):
                if i>1:
                    temp_word_id.append(int(word_id))
#            change the type of  temp_word_id to float64
            temp_word_id_int=temp_word_id
            qid_words[float(word_split[0])]=temp_word_id_int
    return qid_words
    
#this function is used to build embedding_vec,return words_vec is a map
def read_embedding_vec(data_dir,embed_wiki_pdc_d50_norm):
    file_name=os.path.join(data_dir,embed_wiki_pdc_d50_norm)
    with open(file_name,'r') as f:
        lines=f.readlines()
        words_vec=dict()
        for line in lines:
            vec_split=line.split()
            temp_word_vec=[]
#            for word_split,i in word_split:
            for i,vec in enumerate(vec_split):
                if i>0:
                    temp_word_vec.append(float(vec))
#            change the type of  temp_word_id to float64
            temp_word_id_int=temp_word_vec
	    #print('word_vec_name:',vec_split[0])
            words_vec[float(vec_split[0])]=temp_word_id_int
        #print('this is words_vec length:',len(words_vec))
        #this is words_vec length: 96379
    return words_vec

def read_relation(data_dir):
    file_names=[os.path.join(data_dir,'relation.train.fold%d.txt'%i)
    for i in range(0,5)]
    relation=[]
    for file_name in  file_names:
        #print('this is file_name:',file_name)
        if not tf.gfile.Exists(file_name):
            raise ValueError('the file named ',file_name,' is not exist!')
        else:
            with open(file_name,'r') as f:
                lines=f.readlines()
#                count=0
                for line in lines:
#                    if count>10:
#                        break
#                    count+=1
			line_split=line.split()
			temp_split=[]
			temp_split.append(line_split[0])
			temp_split.append(line_split[1])
			temp_split.append(line_split[2])
			relation.append(temp_split)
    return relation
embeddings_vec=read_embedding_vec(data_dir,embed_wiki_pdc_d50)
#print('embeddings:',embeddings_vec)
relation=read_relation(data_dir)
word_qid=read_word_by_qid(data_dir,qid_query)
word_did=read_word_by_did_old(data_dir,docid_doc)

    
def get_batch_doc_vec(relate_doc,not_relate_doc,min_samples,batch_size,shuffle_flag):
	if shuffle_flag==False:
		r_doc_vec,r_query_vec=tf.train.batch(tensors=relate_doc,batch_size=batch_size,num_threads=4,capacity=min_samples+3*batch_size)
		n_doc_vec,n_query_vec=tf.train.batch(tensors=not_relate_doc,batch_size=batch_size,num_threads=4,capacity=min_samples+3*batch_size)    
	else:
		r_doc_vec,r_query_vec=tf.train.shuffle_batch(tensors=relate_doc,batch_size=batch_size,num_threads=4,min_after_dequeue=min_samples,capacity=min_samples+3*batch_size)
		n_doc_vec,n_query_vec=tf.train.shuffle_batch(tensors=not_relate_doc,batch_size=batch_size,num_threads=4,min_after_dequeue=min_samples,capacity=min_samples+3*batch_size)
	return r_doc_vec,r_query_vec,n_doc_vec,n_query_vec

#查找不相关位置的开始节点
def get_not_relate_doc_index(relate_doc_index):
	temp_qid=ralation(relate_doc_index)[1]
	index=relate_doc_index
	for rela in relation[relate_doc_index+1:]:
		index+=1
		if not rela[1]==temp_qid:
			break
	return index
def preprocess_input_data():
	#当前相关联的qid
	now_relate_qid=store_batch_parm.now_relate_qid
	#当前不相关联的qid
	now_not_relate_qid=store_batch_parm.now_not_relate_qid
	#当前相关联的文章的位置
	relate_query_words=word_qid[float(relation[now_relate_qid][1])]
	#print('relate_query:',relate_query_words)
	query_word_length=store_batch_parm.query_word_length
	start_not_relate_doc=store_batch_parm.start_not_relate_doc
	relate_query_word_vec=[]
	for word in relate_query_words:
		#print('word relate_query_vec:',word)
		#print('relate_query_vec:',relate_query_words)
		temp_vec=embeddings_vec.get(word)
		if temp_vec is not None:
			print('temp_vec:',temp_vec)
			relate_query_word_vec.append(embeddings_vec.get(word))
	#如果数据长度大于5,就截取前面5个字符
	now_query_word_length=len(relate_query_word_vec)
	if now_query_word_length >query_word_length:
		relate_query_word_vec=relate_query_word_vec[:query_word_length]
	if now_query_word_length <query_word_length:
		diff_length=query_word_length-now_query_word_length
		#如果长度不足5,将长度添加到5,添加的元素为0
		for _ in range(0,diff_length):
			relate_query_word_vec.append(tf.zeros([len(relate_query_word_vec[0])]))
	#print('relate query and length:',relate_query_words,'    长度是:',len(relate_query_words))
	#if relate_query_word_vec is not None:
                #print('relate_query_word_vec is not None!',relate_query_word_vec,'    长度是:',len(relate_query_word_vec))


	#添加相关或者不相关的词向量
	relate_doc_words=word_did[relation[now_relate_qid][2]]
	not_doc_words=word_did[relation[now_not_relate_qid][2]]
	doc_word_length=store_batch_parm.doc_word_length

	#相关的文章单词向量
        relate_doc_word_vec=[]
        for word in relate_doc_words:
		temp_vec=embeddings_vec.get(word)
                if temp_vec is not None:
                	relate_doc_word_vec.append(embeddings_vec.get(word))
        #如果数据长度大于200,就截取前面200个字符
        now_doc_word_length=len(relate_doc_word_vec)
        if now_doc_word_length >doc_word_length:
                relate_doc_word_vec=relate_doc_word_vec[:doc_word_length]
        if now_doc_word_length <doc_word_length:
                diff_length=now_doc_word_length-doc_word_length
                #如果长度不足200,将长度添加到200,添加的元素为0
                for _ in range(0,diff_length):
                        relate_doc_word_vec.append(tf.zeros([len(relate_doc_word_vec[0])]))
	

	#不相关的文章单词向量
        not_doc_word_vec=[]
        for word in not_doc_words:
		temp_vec=embeddings_vec.get(word)
                if temp_vec is not None:
                	not_doc_word_vec.append(embeddings_vec.get(word))
        #如果数据长度大于5,就截取前面5个字符
        now_doc_word_length=len(not_doc_word_vec)
        if now_doc_word_length >doc_word_length:
                not_doc_word_vec=not_doc_word_vec[:doc_word_length]
        if now_doc_word_length <doc_word_length:
                diff_length=now_doc_word_length-doc_word_length
                #如果长度不足200,将长度添加到200,添加的元素为0
                for _ in range(0,diff_length):
                        not_doc_word_vec.append(tf.zeros([len(not_doc_word_vec[0])]))

	#print
	#print('now_relate_relation:',relation[now_relate_qid])
	#print('now_not_relate_relation:',relation[now_not_relate_qid])
	#遍历相关或者不相关文章时候的逻辑关系
	#判断下一个没有关联的文章的qid是否和有关联的qid是一样的，如果是一样的，则继续下一个，如果不是一样的，则判断没有关联的qid的下一个位置是否已经结束，若没有，返回到开始的不相关的qid的位置，如果结束了，则查找下一个qid不相关的文章的开始位置
	#print
	#print('relate_query:',relate_query_word_vec)
	#print('relate_relate_doc:',relate_doc_word_vec)
	#print('relate_not_relate_doc:',not_doc_word_vec)
	#if relate_doc_word_vec is not None:
		#print('relate_doc_word_vec is not None!',not_doc_word_vec)
	if not relation[now_not_relate_qid+1][1]==relation[now_not_relate_qid][1]:
		if not relation[now_relate_qid+1][0]==relation[now_relate_qid][0]:
			now_relate_qid=now_relate_qid+1
			now_not_relate_qid=get_not_relate_doc_index(now_relate_qid)
			start_not_relate_doc=now_not_relate_qid
		else:
			now_relate_qid+=1
                        now_not_relate_qid=get_not_relate_doc_index(now_relate_qid)
	else:
		now_not_relate_qid+=1
	min_samples=int(0.4*5000)
	print('relate_query_words:',relate_query_words)
	#r_doc_vec,r_query_vec,n_doc_vec,n_query_vec=get_batch_doc_vec([relate_doc_word_vec,relate_query_word_vec],[not_doc_word_vec,relate_query_word_vec],min_samples,batch_size,False)
	#r_doc_vec=tf.reshape(tensor=r_doc_vec,shape=(batch_size,doc_word_length,word_vec_length,1))
	#r_query_vec=tf.reshape(tensor=r_query_vec,shape=(batch_size,query_word_length,word_vec_length,1))
	#n_doc_vec=tf.reshape(tensor=n_doc_vec,shape=(batch_size,doc_word_length,word_vec_length,1))
	#n_query_vec=tf.reshape(tensor=n_query_vec,shape=(batch_size,query_word_length,word_vec_length,1))
	return relate_doc_word_vec,not_doc_word_vec,relate_query_word_vec
#preprocess_input_data()
#preprocess_input_data()

def get_batch_data(batch_size):
	relate_batch_data=[]
	not_batch_data=[]
	query_batch_data=[]
	for i in range(0,batch_size):
		relate_doc_word_vec,not_doc_word_vec,relate_query_word_vec=preprocess_input_data()
		relate_batch_data.append(relate_doc_word_vec)
		not_batch_data.append(not_doc_word_vec)
		query_batch_data.append(relate_query_word_vec)
	print('length_r_n_q:',len(relate_batch_data),len(not_batch_data),len(query_batch_data))
	r_doc_vec=tf.reshape(tensor=relate_batch_data,shape=(batch_size,doc_word_length,word_vec_length,1))
        r_query_vec=tf.reshape(tensor=tf.cast(query_batch_data,tf.float32),shape=(batch_size,query_word_length,word_vec_length,1))
        n_doc_vec=tf.reshape(tensor=not_batch_data,shape=(batch_size,doc_word_length,word_vec_length,1))
	return r_doc_vec,n_doc_vec,r_query_vec

get_batch_data(batch_size)


	
