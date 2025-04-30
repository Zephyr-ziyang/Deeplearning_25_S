# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import os 

# fixed seed for experiment
np.random.seed(309)

# 加载MNIST训练数据
base_dir = os.path.dirname(os.path.abspath(__file__))
train_images_path = os.path.join(base_dir, 'dataset', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(base_dir, 'dataset', 'MNIST', 'train-labels-idx1-ubyte.gz')

# 读取图像和标签数据
with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 随机划分验证集(10000个样本)
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 归一化, from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

model_type = 'CNN'
if model_type == 'MLP':
    model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
elif model_type == 'CNN':
    # 注意CNN需要调整输入形状
    train_imgs = train_imgs.reshape(-1, 1, 28, 28)  # 转换为CNN需要的形状 [N, C, H, W]
    valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
    model = nn.models.Model_CNN(input_shape=(1, 28, 28), num_classes=10)


optimizer = nn.optimizer.SGD(init_lr=0.01, model=model)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, batch_size=128)

# print("初始参数示例:")
# for i, param in enumerate(linear_model.parameters()):
#     if i == 0:  # 只打印第一层的参数作为示例
#         print(param[:5])  # 打印前5个参数
#         break


runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=os.path.join(base_dir, 'best_models'))

# # 训练后打印参数
# print("\n训练后参数示例:")
# for i, param in enumerate(linear_model.parameters()):
#     if i == 0:
#         print(param[:5])  # 打印前5个参数
#         break

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()