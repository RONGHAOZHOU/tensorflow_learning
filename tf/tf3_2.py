import tensorflow as tf
# x = tf.constant([[1.0, 2.0]])
# w = tf.constant([[3.0], [4.0]])
#
# x=tf.constant([[2,3]])
# w=tf.constant([[6],[7]])
x=tf.constant([[2,4,6],[9,5,4],[9,7,4]]) #三行三列矩阵
w=tf.constant([[3,2],[6,6],[9,4]])  #三行二列矩阵


y=tf.matmul(x,w) #矩阵乘法
print (y)
with tf.Session() as sess:
    print (sess.run(y))


