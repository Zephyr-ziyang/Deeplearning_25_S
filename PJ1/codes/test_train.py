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

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])

optimizer = nn.optimizer.SGD(init_lr=0.01, model=linear_model)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, batch_size=128)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=100, save_dir=r'./best_models')


_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()