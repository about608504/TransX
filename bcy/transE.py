#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")
test_lib = ll("./test.so")

class Config(object):

	def __init__(self):
		lib.setInPath("")
		test_lib.setInPath("")
		lib.setBernFlag(0)
		self.learning_rate = 0.001
		self.testFlag = False
		self.loadFromData = True
		self.L1_flag = True
		self.hidden_size = 100
		self.nbatches = 100
		self.entity = 0
		self.relation = 0
		self.trainTimes = 1000
		self.margin = 1.0

class TransEModel(object):

	def __init__(self, config):

		entity_total = config.entity
		relation_total = config.relation
		batch_size = config.batch_size
		size = config.hidden_size
		margin = config.margin

		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])
		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

		with tf.name_scope("embedding"):
			self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			#在embedding中进行查找
			pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
			pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
			pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
			neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
			neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
			neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

		if config.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
			self.predict = pos
		else:
			#压缩求和，用于降维，L2是平方
			pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
			neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
			self.predict = pos

		with tf.name_scope("output"):
			self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))


def main(_):
	config = Config()
	if (config.testFlag):
		test_lib.init()
		config.relation = test_lib.getRelationTotal()
		config.entity = test_lib.getEntityTotal()
		config.batch = test_lib.getEntityTotal()
		config.batch_size = config.batch
	else:
		lib.init()
		config.relation = lib.getRelationTotal()
		config.entity = lib.getEntityTotal()
		config.batch_size = lib.getTripleTotal() // config.nbatches
	
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			#tf.contrib.layers.xavier_initializer此函数实现权重初始化：
			#初始化器旨在使所有层中的梯度比例保持大致相同。在均匀分布中，这最终是范围：
			#  x = sqrt(6. / (in + out)); [-x, x]并且对于正态分布，使用标准偏差sqrt(2. / (in + out))。
			initializer = tf.contrib.layers.xavier_initializer(uniform = False)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				trainModel = TransEModel(config = config)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			#基本的优化类，该类不常常被直接调用，而较多使用其子类，
			# 比如GradientDescentOptimizer, AdagradOptimizer或者MomentumOptimize
			optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
			#对var_list中的变量计算loss的梯度
			# 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
			grads_and_vars = optimizer.compute_gradients(trainModel.loss)
			#将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			saver = tf.train.Saver()
			sess.run(tf.initialize_all_variables())
			if (config.loadFromData):
				saver.restore(sess, 'model.vec')

			def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
					trainModel.neg_h: neg_h_batch,
					trainModel.neg_t: neg_t_batch,
					trainModel.neg_r: neg_r_batch
				}
				_, step, loss = sess.run(
					[train_op, global_step, trainModel.loss], feed_dict)
				return loss

			def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
				}
				step, predict = sess.run(
					[global_step, trainModel.predict], feed_dict)
				return predict

			ph = np.zeros(config.batch_size, dtype = np.int32)
			pt = np.zeros(config.batch_size, dtype = np.int32)
			pr = np.zeros(config.batch_size, dtype = np.int32)
			nh = np.zeros(config.batch_size, dtype = np.int32)
			nt = np.zeros(config.batch_size, dtype = np.int32)
			nr = np.zeros(config.batch_size, dtype = np.int32)

			ph_addr = ph.__array_interface__['data'][0]
			pt_addr = pt.__array_interface__['data'][0]
			pr_addr = pr.__array_interface__['data'][0]
			nh_addr = nh.__array_interface__['data'][0]
			nt_addr = nt.__array_interface__['data'][0]
			nr_addr = nr.__array_interface__['data'][0]

			lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
			test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
			test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
			test_lib.testHead.argtypes = [ctypes.c_void_p]
			test_lib.testTail.argtypes = [ctypes.c_void_p]

			if not config.testFlag:
				for times in range(config.trainTimes):
					res = 0.0
					for batch in range(config.nbatches):
						lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
						res += train_step(ph, pt, pr, nh, nt, nr)
						current_step = tf.train.global_step(sess, global_step)
					print(times)
					print(res)
				saver.save(sess, './model.vec')
			else:
				total = test_lib.getTestTotal()
				for times in range(total):
					test_lib.getHeadBatch(ph_addr, pt_addr, pr_addr)
					res = test_step(ph, pt, pr)
					test_lib.testHead(res.__array_interface__['data'][0])

					test_lib.getTailBatch(ph_addr, pt_addr, pr_addr)
					res = test_step(ph, pt, pr)
					test_lib.testTail(res.__array_interface__['data'][0])
					print(times)
					if (times % 50 == 0):
						test_lib.test()
				test_lib.test()

if __name__ == "__main__":
	tf.app.run()
