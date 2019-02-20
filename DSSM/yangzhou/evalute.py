#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import forward_prop
import numpy as np
#训练的每个周期的大小
train_samples_per_epoch=forward_prop.train_samples_per_epoch
#测试的每个周期的大小
test_samples_per_epoch=forward_prop.test_samples_per_epoch
#batch_size
batch_size=store_batch_parm.batch_size

def eval_once(summary_op,summary_writer,saver,predict_true_or_false):
	with tf.Session() as sess:
		checkpoint_proto=tf.train.get_checkpoint_state(checkpoint_dir=train.check.point_path)
		if checkpoint_proto and checkpoint_proto.model_checkpoint_path:
			#恢复变量到模型中
			saver.restore(sess,checkpoint_proto.model_checkpoint_path)
		else:
			print('checkpoint files is not found')
			return
		coord=tf.train.Coordinator()
		try:
			threads=[]
			for queue_runner in tf.get_collection(key=tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(queue_runner.create_threads(sess,coord=coord,daemon=True,start=True))
				#测试块个数，向上取整
				test_batch_num=math.ceil(test_samples_per_epoch/batch_size)
				iter_num=0
				true_test_num=0
				total_test_num=test_batch_num*batch_size
				while iter_num<test_batch_num and not coord.should_stop():
					return_judge=sess.run([predict_true_or_false])
					true_test_num+=np.sum(return_judge)
					iter_num+=1
				precision=true_test_num/total_test_num
				print('the test precision is %3.f'%precision)
		except:
			coord.request_stop()
		coord.request_stop()
		coord.join(threads)
#设置空位符
query_vec=tf.placeholder('float',shape=[-1,query_word_length,word_vec_length，1])
r_doc_vec=tf.placeholder('float',shape=[-1,doc_word_length,word_vec_length，1])
n_doc_vec=tf.placeholder('float',shape=[-1,doc_word_length,word_vec_length，1])


def evaluate():
	with tf.Graph().as_default() as g:
		relate_doc_word_vec,not_doc_word_vec,relate_query_word_vec=test_file.get_batch_data()
		loss=test_file.network(query_vec,r_doc_vec,n_doc_vec,drop_out=False)
		predict_true_or_false=False
		if loss==0:
			predict_true_or_false=True
		#创建衰减对象
		moving_average_op=tf.train.ExponentialMovingAverage(decay=test.moving_average_decay)	
		variables_to_restore=moving_average_op.variables_to_restore()
		#创建还原对象
		saver=tf.train.Saver(var_list=variables_to_restore)
		#创建序列化之后的summary对象
		summary_op=tf.summary.merage_all_summaries()
		summary_writer=tf.summary.FileWriter(logdir='./event-log-test',graph=g)
		eval_once(summary_op,summary_writer,saver,predict_true_or_false)















