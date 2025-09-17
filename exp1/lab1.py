import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore import Model, dataset as ds
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig

import os
import numpy as np
from sklearn.model_selection import train_test_split


# 加载数据集
from numpy import genfromtxt

script_dir = os.path.dirname(os.path.abspath(__file__))
iris_file_path = os.path.join(script_dir, 'iris.csv')
iris_data = genfromtxt(iris_file_path, delimiter=',')
print(iris_data[:10])

iris_data = iris_data[1:]
X = iris_data[:, :4].astype(np.float32)
y = iris_data[:, -1].astype(np.int32)

X /= np.max(np.abs(X), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = (X_train, y_train)
train_data = ds.NumpySlicesDataset(train_data)

test_data = (X_test, y_test)
test_data = ds.NumpySlicesDataset(test_data)

train_data = train_data.batch(32)
test_data = test_data.batch(32)


# 定义网络
class my_net(nn.Cell):
    def __init__(self):
        super(my_net, self).__init__()
        self.fc1 = nn.Dense(4, 10)
        self.fc2 = nn.Dense(10, 3)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 训练模型
net = my_net()

net_loss = SoftmaxCrossEntropyWithLogits(sparse=True)

lr = 0.01
momentum = 0.9
net_opt = nn.Momentum(net.trainable_params(), lr, momentum)

model = Model(net, net_loss, net_opt, metrics={"accuracy": Accuracy()})

# 配置检查点回调
config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=3)
ckpoint = ModelCheckpoint(prefix="iris_model", config=config_ck)


# 使用LossMonitor回调函数来输出训练日志
# LossMonitor(per_print_times=1) 表示每1个batch打印一次loss
print("\n开始训练模型...")
model.train(10, train_data, callbacks=[LossMonitor(per_print_times=1), ckpoint])

eval_result = model.eval(test_data)

print("\n评估结果:")
print(f"准确率: {eval_result['accuracy']:.4f}")
