from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p=0.0, patience=5):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_p = dropout_p
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list)-1):
                # 添加线性层
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i+1])
                if lambda_list and i < len(lambda_list):
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                
                # 在隐藏层后添加激活函数和Dropout
                if i < len(size_list)-2 and act_func == 'ReLU':
                    self.layers.append(ReLU())
                    if dropout_p > 0:
                        self.layers.append(Dropout(p=dropout_p))
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
    
    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def early_stopping(self, val_loss):
        """Early stopping机制"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
    
        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
    def parameters(self):
        """返回所有可训练参数"""
        params = []
        for layer in self.layers:
            if layer.optimizable and hasattr(layer, 'params'):
                params.extend([layer.params['W'], layer.params['b']])  # 分别添加W和b
        return params
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
                
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        

class Model_CNN(Layer):
    """
    A model with conv2D layers.
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10): #默认输入形状为Dataset的形状
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 计算卷积层输出形状
        self.conv_output_shape = (
            32,  # 通道数
            input_shape[1]//4,  # 高度 (经过两次2x2池化)
            input_shape[2]//4   # 宽度 (经过两次2x2池化)
        )

        # 定义网络结构
        self.conv1 = conv2D(in_channels=input_shape[0], out_channels=16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        
        # 计算全连接层输入尺寸
        conv_output_size = 32 * (input_shape[1]//4) * (input_shape[2]//4)
        self.fc1 = Linear(in_dim=conv_output_size, out_dim=128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_dim=128, out_dim=num_classes)
        
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc1, self.relu3, self.fc2]


    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 确保输入尺寸正确
        assert X.shape[1:] == self.input_shape
        
        # 卷积层 Conv1 + ReLU 激活
        x = self.conv1(X)
        x = self.relu1(x)
        # 记录池化位置
        self.pool1_mask = np.zeros_like(x)
        h, w = x.shape[2], x.shape[3]
        self.pool1_mask[:, :, ::2, ::2] = 1
        x = x[:, :, ::2, ::2]  # 最大池化
        
        # 卷积层 Conv2 + ReLU 激活
        x = self.conv2(x)
        x = self.relu2(x)
        # 记录池化位置
        self.pool2_mask = np.zeros_like(x)
        self.pool2_mask[:, :, ::2, ::2] = 1
        x = x[:, :, ::2, ::2]  # 最大池化
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # FC层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

    def backward(self, loss_grad):
        # 全连接层反向传播
        grads = self.fc2.backward(loss_grad)
        grads = self.relu3.backward(grads)
        grads = self.fc1.backward(grads)
        
        # 恢复池化前的形状
        grads = grads.reshape(-1, 32, 7, 7)
        # 最大池化反向传播
        pool2_grad = np.zeros((grads.shape[0], 32, 14, 14))
        pool2_grad[:, :, ::2, ::2] = grads
        grads = pool2_grad * self.pool2_mask
        
        grads = self.relu2.backward(grads)
        grads = self.conv2.backward(grads)
        
        # 第一个池化层反向传播
        pool1_grad = np.zeros((grads.shape[0], 16, 28, 28))
        pool1_grad[:, :, ::2, ::2] = grads
        grads = pool1_grad * self.pool1_mask
        
        grads = self.relu1.backward(grads)
        grads = self.conv1.backward(grads)
        
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        # 加载模型参数
        for i, layer in enumerate(self.layers):
            if layer.optimizable:
                layer.W = param_list[i]['W']
                layer.b = param_list[i]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                if hasattr(layer, 'weight_decay'):
                    layer.weight_decay = param_list[i]['weight_decay']
                    layer.weight_decay_lambda = param_list[i]['lambda']

        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_dict = {
                    'W': layer.params['W'],
                    'b': layer.params['b']
                }
                if hasattr(layer, 'weight_decay'):
                    param_dict['weight_decay'] = layer.weight_decay
                    param_dict['lambda'] = layer.weight_decay_lambda
                param_list.append(param_dict)
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
                
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False