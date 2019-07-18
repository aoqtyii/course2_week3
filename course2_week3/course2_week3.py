import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

# np.random.seed(1)
#
# y_hat = tf.constant(36, name="y_hat")
# y = tf.constant(39, name="y")
# loss = tf.Variable((y-y_hat)**2, name="loss")
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))
#
#     x = tf.placeholder(tf.int64, name="x")
#     print(session.run(x*2, feed_dict={x: 3}))
#     session.close()

# 线性函数
def linear_function():
    np.random.seed(1)

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)
    session = tf.Session()
    result = session.run(Y)

    session.close()

    return result

# print("result = " + str(linear_function()))

# 计算sigmoid
def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")
    sigmoid = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(sigmoid, feed_dict={x: z})
    return result
# print("sigmoid(0) = " + str(sigmoid(0)))
# print("sigmoid(12) = " + str(sigmoid(12)))

# 独热编码
def one_hot_matrix(labels, C):
    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    with tf.Session() as session:
        one_hot = session.run(one_hot_matrix)
    return one_hot
# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels, C=4)
# print(str(one_hot))

# 初始化0,1
def ones(shape):
    ones = tf.ones(shape)
    with tf.Session() as session:
        ones = session.run(ones)
    return ones
# print("ones = " + str(ones([3])))

# 使用tensorflow搭建神经网络
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, calsses = tf_utils.load_dataset()

# index = 12
# plt.imshow(X_train_orig[index])
# plt.show()
# print("Y = " + str(np.squeeze(Y_train_orig[:, index])))

# 数据扁平化
# print("x_train_orig" + str(X_train_orig.shape))
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# print("X_train_flatten.shape = " + str(X_train_flatten.shape))
# print("X_test_flatten.shape = " + str(X_test_flatten.shape))

# 归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# 转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)

# print("训练集样本数 = " + str(X_train.shape[1]))
# print("测试集样本数 = " + str(Y_test.shape[1]))
# print("X_train.shape = " + str(X_train.shape))
# print("Y_train.shape = " + str(Y_train.shape))
# print("X_test.shape = " + str(X_test.shape))
# print("Y_test.shape = " + str(Y_test.shape))

# 为tensorflow会话创建占位符
def create_placeholders(n_x, n_y):
    """
    为tensorflow会话创建占位符
    :param n_x: 一个实数，图片向量的大小(64*64*3 = 12288)
    :param n_y: 一个实数，分类数(从0到5，所以n_y = 6)
    :return: X - 一个数据输入的占位符，维度为[n_x, None], dtype = "float"
             Y - 一个对应输入的标签占位符，维度为[n_Y, None], dtype = "float"
    """
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name= "Y")

    return X, Y
# # 测试
# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))

# 初始化参数
def initialize_parameters():
    """
    初始化神经网络的参数，参数的维度如下
    W1: [25,12288]
    b1: [25,1]
    W2: [12,25]
    b2: [12,1]
    W3: [6,12]
    b3: [6,1]
    :return: parameters 包含了W和b的字典
    """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    return parameters
# # 测试
# tf.reset_default_graph #用于清除默认图形堆栈并重置全局默认图形
#
# with tf.Session() as session:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))
#     print("W3 = " + str(parameters["W3"]))
#     print("b3 = " + str(parameters["b3"]))

# 向前传播
def forward_propagation(X, parameters):
    # 字典赋值不连写
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    return Z3
# # 测试
# tf.reset_default_graph()
# with tf.Session() as session:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print("Z3 = " + str(Z3))

# 计算成本
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost
# # 测试
# tf.reset_default_graph()
# with tf.Session() as session:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))

# 反向传播 更新参数：仅用一行代码执行
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 构建模型
def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=True):
    ops.reset_default_graph() # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    # 反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 初始化所有的变量
    init = tf.global_variables_initializer()
    # 开始会话并计算
    with tf.Session() as session:
        session.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost = epoch_cost + minibatch_cost/num_minibatches

            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "  epoch_cost = " + str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.xlabel('iterations')
            plt.ylabel('cost')
            plt.title('learning_rate = ' + str(learning_rate))
            plt.show()

        parameters = session.run(parameters)
        print("参数已经保存到session中")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率：", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters

# 测试
start_time = time.clock()
parameters = model(X_train, Y_train, X_test, Y_test)
end_time = time.clock()
print("start testing")
print("CPU执行时间 = " + str(end_time-start_time) + "秒")
