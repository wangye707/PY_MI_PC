#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import store_batch_parm
#训练的每个周期的大小
train_samples_per_epoch=5000
#测试的每个周期的大小
test_samples_per_epoch=1000

# 定义数据集所在文件夹路径  
data_dir='./cifar-10-batches-bin'
#each time train batch
#用于移动平均线的衰减
moving_average_decay=0.9999
# 衰减呈阶梯函数，控制衰减周期（阶梯宽度）
num_epochs_per_decay=350.0
# 学习率衰减因子 
learning_rate_decay_factor=0.1
# 初始学习率
initial_learning_rate=0.1
#batch_size
batch_size=store_batch_parm.batch_size

def variable_on_gpu(name,shape,dtype,initializer):
	with tf.device('/gpu:0'):
		return tf.get_variable(name=name,shape=shape,initializer=initializer,dtype=dtype)

def variable_on_gpu_with_collection(name,shape,dtype,stddev,wd):
	with tf.device('gpu:0'):
		weight=tf.getvariable(name=name,shape=shape,initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
		if wd is not None:
			tf.add_tocollection(name='losses',value=weight_decay)
		return weight

def losses_summary(total_loss):
	#创建滑动指数平均对象,此对象用户更新滑动平均参数，decay用于更新模型的速度
	average_op=tf.train.ExponentialMovingAverage(decay=0.9)
	#从字典集合中返回losses对象集合
	losses=tf.collection(key='lossse')
	#加入更新列表
	matin_average_op=average_op.apply(losses+[total_loss])
	for i in losses+[total_loss]:
		#将变量写道summary缓存对象
		tf.contrib.deprecated.sclar_summary(i.op.name+'_raw')
		tf.contrib.deprecated.sclar_summary(i.op.name,average_op.average(i))
	return matin_average_op

def one_step_train(total_loss,step):
	#每个周期所需要的batch_size个数
	batch_count=int(train_samples_per_epoch/batch_size)
	#衰减步
	decay_step=batch_count*num_epochs_per_decay
	#对学习率进行衰减
	lr=tf.train.exponential_decay(learning_rate=initial_learning_rate,global_step=step,decay_steps=decay_step,decay_rate=learning_rate_decay_factor,staircase=True)
	tf.contrib.deprecated.scalar_summary('learning_rate',lr)
	losses_movingaverage_op=loss_summary(total_loss)
	#创建语境管理器
	with tf.control_dependencies(control_inputs=[losses_movinaverage_op]):
		trainer=tf.train.GradientDescentOptimizer(learning_rate=lr)
		#计算梯度更新，并返回一个键值对列表，默认更新var_list=GraphKey.TRAINABLE_VARIABLES
		gradient_pairs=trainer.compute_gradients(loss=total_loss)
	#进一步更新操作
	gradient_update=trainer.apply_gradients(grads_and_vars=gradient_pairs,global_step=step)
	variables_average_op=tf.train.ExponentialMovingAverage(decay=moving_average_decay,num_updates=step)
	maintain_variables_op=variables_average_op.apply(var_list=tf.trainable_variables())
		
	with tf.control_dependencies(control_inputs=[gradient_update,maintain_variables_op]):
		gradient_update_optimizer=tf.no_op()
	return gradient_update_optimizer

		
def network(query_vec,relate_doc,not_relate_doc,drop_out=True):
	with tf.variable_scope(name_or_scope='conv1') as scope:
		q_weight=variable_on_gpu_with_collection(name='q_weight',shape=(1,5,1,32),dtype=tf.float32,stddev=0.05,wd=0.0)
		q_bias=variable_on_gpu_with_collection(name='q_bias',shape=(32),dtype=tf.float,initializer=tf.constant_initializer(value=0.0))
		
		d_weight=variable_on_gpu_with_collection(name='d_weight',shape=(20,5,1,32),dtype=tf.float32,stddev=0.05,wd=0.0)
                d_bias=variable_on_gpu_with_collection(name='d_bias',shape=(32),dtype=tf.float,initializer=tf.constant_initializer(value=0.0))
		#乘上权重，卷积操作
		conv1_query_in=tf.nn.con2d(input=query_vec,filter=q_weight,strides=(1,1,5,1))
		conv1_relate_in=tf.nn.con2d(input=relate_doc,filter=d_weight,strides=(1,20,5,1))
		conv1_not_relate_in=tf.nn.con2d(input=not_relate_doc,filter=d_weight,strides=(1,20,5,1))
		#加入偏置量
		conv1_query_in=tf.nn.bias_add(value=conv1_query_in,bias=q_bias)
		conv1_relate_in=tf.nn.bias_add(value=conv1_relate_in,bias=d_bias)
		conv1_not_relate_in=tf.nn.bias_add(value=conv1_not_relate_in,bias=d_bias)
		#tanh激活层
		conv1_query_in=tf.nn.tanh(conv1_query_in)
		conv1_relate_in=tf.nn.tanh(conv1_relate_in)
		conv1_not_relate_in=tf.nn.tanh(conv1_not_relate_in)
		#为防止过拟合，加入dropOut
		if drop_out is True:
			keep_prob=tf.placeholder('float')
			conv1_query_in=tf.nn.dropout(conv1_query_in,keep_prob)
			conv1_relate_in=tf.nn.dropout(conv1_relate_in,keep_prob)
			conv1_not_relate_in=tf.nn.dropout(conv1_not_relate_in,keep_prob)
		#池化层
		conv1_query_pool=tf.nn.max_pool(value=conv1_query_in,ksize=(1,2,2,1),strides=(1,1,2,1),padding='SAME')
		conv1_relate_pool=tf.nn.max_pool(value=conv1_relate_in,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
		conv1_not_relate_pool=tf.nn.max_pool(value=conv1_not_relate_in,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
		#局部归一化
		norm1_query_pool=tf.nn.lrn(input=conv1_query_pool,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)
		norm1_relate_pool=tf.nn.lrn(input=conv1_relate_pool,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)
		norm1_not_relate_pool=tf.nn.lrn(input=conv1_not_relate_pool,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)
		#flatten，压平向量，然后相乘
		#flatten_query=[y for x in conv1_query_pool for y in x] 
		#flatten_relate=[y for x in conv1_relate_pool for y in x] 
		#flattent_not=[y for x in conv1_not_relate_pool for y in x]
		reshape_query=tf.reshape(tensor=norm1_query_pool,shape=(-1,5*5*32))
		reshape_relate=tf.reshape(tensor=norm1_query_pool,shape=(-1,5*5*32))
		reshape_not_relate=tf.reshape(tensor=norm1_query_pool,shape=(-1,5*5*32))
	with tf.variable_scope(name_or_scopt='flatten1') as scope:
		q_weight=variable_on_gpu_with_collection(name='q_weight',shape=(5*5*32,32),dtype=tf.float32,stddev=0.04,wd=0.004)
                q_bias=variable_on_gpu_with_collection(name='q_bias',shape=(32),dtype=tf.float,initializer=tf.constant_initializer(value=0.0))

                d_weight=variable_on_gpu_with_collection(name='d_weight',shape=(5*5*32,32),dtype=tf.float32,stddev=0.04,wd=0.004)
                d_bias=variable_on_gpu_with_collection(name='d_bias',shape=(32),dtype=tf.float,initializer=tf.constant_initializer(value=0.0))
		#全链接层
		layer_query=tf.matmul(reshape_query,q_weight)+q_bias
		layder_relate=tf.matmul(reshape_relate,d_weight)+d_bias
		layder_not_relate=tf.matmul(reshape_not_relate,d_weight)+d_bias	
		#计算两个向量之间的夹角，取绝对值
		relate_multiply=tf.multiply(layer_query,layder_relate)
		not_relate_multiply=tf.multiply(layer_query,layder_not_relate)
		relate_multiply=tf.abs(relate_multiply)
		not_relate_multiply=tf.abs(not_relate_multiply)
		#每个向量自己相乘之后的值
		sum_query=tf.sqrt(tf.reduce_sum(tf.square(layer_query)))
		sum_relate=tf.sqrt(tf.reduce_sum(tf.square(layder_relate)))
		sum_not_relate=tf.sqrt(tf.reduce_sum(tf.square(layder_not_relate)))
		#计算query与document之间的夹角
		cos_relate=sum_relate/sum_query
		cos_not_relate=sum_not_relate/sum_query
		#计算损失
		loss=tf.maximum(0,1-cos_relate+cos_not_relate)
	#tf.add_n：把一个列表的东西都依次加起来,将所有的损失加起来
	return tf.add_n(inputs=tf.get_collection(key='losses'),name='total_loss')








