import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 学习率
learning_rate = 0.01
# 最大训练步数
max_train_steps = 1000
log_step = 50

# 构造训练数据
train_X = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182],
                    [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [5.654],
                    [9.27], [3.1]], dtype=np.float32)

train_Y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596],
                    [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [2.42], [2.94], [1.3]],
                   dtype=np.float32)

total_samples = train_X.shape[0]

# 输入数据
X = tf.placeholder(tf.float32, [None, 1])

# 模型参数
W = tf.Variable(tf.random_normal([1, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 推理值
Y = tf.matmul(X, W) + b

# 实际值
Y_ = tf.placeholder(tf.float32, [None, 1])

# 均方差
loss = tf.reduce_sum(tf.pow(Y-Y_, 2)) / (total_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    print("Start training:")
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_:train_Y})
        if step % log_step == 0:
            # 每隔log_step步打印一次日至
            c = sess.run(loss, feed_dict={X: train_Y, Y_:train_Y})
            print("Step:%d, loss=%.4f, W=%.4f, b=%.4f" % (step, c, sess.run(W), sess.run(b)))
    # 计算训练完毕的模型在训练集上的损失值，并将其作为指标输出
    final_loss = sess.run(loss, feed_dict={X:train_X, Y_:train_Y})
    # 计算训练完毕的模型参数W和b
    weight, bias = sess.run([W, b])
    print("Step:%d, loss=%.4f, W=%.4f, b=%.4f" % (max_train_steps, final_loss, sess.run(W), sess.run(b)))
    print("Linear Regression Model: Y=%.4f*X+%.4f" % (weight, bias))

    # 根据训练数据X和Y，添加对应的红色圆点
    plt.plot(train_X, train_Y, 'ro', label='Training data')
    # 根据模型参数和训练数据，添加蓝色（默认色）拟合直线
    plt.plot(train_X, weight*train_X+bias, label='Fitted line')
    # 添加图例说明
    plt.legend()
    # 绘制图形
    plt.show()










