# encoding:utf-8
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

# 模拟数据集
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)

Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print('X:', X)
print('Y_:', Y_)

# 1定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], 0.0, 1.0, tf.float32, 1))
w2 = tf.Variable(tf.random_normal([3, 1], 0.0, 1.0, tf.float32, 1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 2定义损失函数和前向传播方法。
loss_mse = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
# train_step = tf.train.AdamOptimizer().minimize(loss_mse)
# train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)

# 3生成会话，训练STEPS轮

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出未经训练的参数取值
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")

    # 训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_lose = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training steps,loss_mes on all data is %g" % (i, total_lose))
    # 输出训练后的参数
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
