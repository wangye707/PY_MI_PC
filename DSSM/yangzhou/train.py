#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import os
import numpy as np

import read_files_new
import store_batch_parm
#设置参数迭代次数
max_iter_num=6000
#设计模型参数所在路径
data_dir=read_files_new.data_dir
checkpoint_path='checkpoint'
#设计事件文件所在路径，用于周期性的summary对象
event_log_path='event-log'
#每组的长度
batch_size=store_batch_parm.batch_size
#文章单词个数
doc_word_length=store_batch_parm.doc_word_length
#查询单词个数
query_word_length=store_batch_parm.query_word_length
#单词向量长度
word_vec_length=store_batch_parm.word_vec_length


def train():
	#制定默认的图
	with tf.Graph().as_default():
		#设置空位符
		query_vec=tf.placeholder('float',shape=[-1,query_word_length,word_vec_length，1])
		r_doc_vec=tf.placeholder('float',shape=[-1,doc_word_length,word_vec_length，1])
		n_doc_vec=tf.placeholder('float',shape=[-1,doc_word_length,word_vec_length，1])
		#设置全局步,trainable防止滑动的过程中对global_step也进行更新
		global_step=tf.Variable(initial_value=0,trainable=False)
		#r_doc_vec,n_doc_vec,query_vec=get_batch_date(batch_size)
		#加入神经网络训练并计算出损失
		total_loss=network(query_vec,r_doc_vec,n_doc_vec)
		#进一步返回梯度更新操作
		one_step_gradient_update=forward_prop.one_step_train(total_loss,global_step)
		#保存模型变量，用于恢复
		saver=tf.train.Saver(var_list=tf.all_variables()
		#tensorBoard输出等级
		all_summary_obj=tf.contrib.deprecated.merge_all_summaries()
		#初始化所有参数
		initiate_variables=tf.initialize_all_variables()
		
		#log_device_placement记录每个使用设备的参数
		with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
			#变量初始化
			sess.run(initiate_variables)
			#创建事件写入对象
			Event_writer=tf.summary.FileWriter(logdir=event_log_path,grap=sess.graph)
			
			for step in range(max_iter_num):
				r_doc_vec,n_doc_vec,query_vec=get_batch_date(batch_size)
				sess.run(feed_dict={query_vec:query_vec,r_doc_vec:r_doc_vec,n_doc_vec:n_doc_vec})
				_,loss_value=sess.run(fetchs=[one_step_gradient_update,total_loss])
				assert not np.isnan(loss_value)
				if step%10==0:
					print('step%d,the loss_value is %.2f'%(step,loss_value))
				if step%100==0:
					all_summaries=sess.run(all_summary_obj)
					Event_writer.add_summary(summary=all_summaries,global_step=step)
				if step%1000==0 or (step+1)==max_iter_num:
					variables_save_path=os.path.join(checkpoint_path,'model-parameters.bin')
					saver.save(sess,variables_save_path,global_step=step)


if __name__=='__main__':
	train()






