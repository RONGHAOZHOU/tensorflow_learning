#coding:utf-8
import tensorflow as tf

#定义输入参数和列表
x=tf.constant([[0.5,0.7]])
w1=tf.Variable(tf.random_normal([2,3],mean=0,stddev=2,seed=1))
w2=tf.Variable(tf.random_normal([3,1],mean=0,stddev=1,seed=1))

#定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(y)
    print("y  result is :",sess.run(y))
