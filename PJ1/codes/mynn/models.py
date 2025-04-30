from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

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
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers.
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10): #默认输入形状为Dataset的形状
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
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
        assert X.shape[1:] == self.input_shape, f"Input shape {X.shape[1:]} doesn't match model's input shape {self.input_shape}"
        
        # 卷积层 Conv1 + ReLU 激活
        x = self.conv1(X)
        x = self.relu1(x)
        # MaxPooling (简单实现最大池化)
        x = x[:, :, ::2, ::2]
        
        # 卷积层 Conv2 + ReLU 激活
        x = self.conv2(x)
        x = self.relu2(x)
        # MaxPooling
        x = x[:, :, ::2, ::2]
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # FC layer
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
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