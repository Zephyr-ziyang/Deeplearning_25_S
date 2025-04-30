from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # 保存输入用于反向传播

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 保存输入用于反向传播
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # 计算W的梯度
        self.grads['W'] = np.dot(self.input.T, grad)
        # 计算b的梯度
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        
        # 如果有权重衰减
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        # 计算传递给前一层的梯度
        output_grad = np.dot(grad, self.W.T)
        return output_grad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # 初始化权重 [out_channels, in_channels, kH, kW]
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = initialize_method(size=(out_channels, 1))
        
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """

        self.input = X
        batch_size, _, H, W = X.shape
        
        # 计算输出尺寸
        new_H = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        new_W = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1
        
        # 添加padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            X_padded = X
            
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))
        
        # 简单实现卷积操作
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(new_H):
                    for w in range(new_W):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # 提取当前窗口
                        window = X_padded[b, :, h_start:h_end, w_start:w_end]
                        # 计算卷积
                        output[b, oc, h, w] = np.sum(window * self.W[oc]) + self.b[oc]
        
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, H, W = self.input.shape
        _, _, new_H, new_W = grads.shape
        
        # 初始化梯度
        grad_input = np.zeros_like(self.input)
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        
        # 添加padding
        if self.padding > 0:
            X_padded = np.pad(self.input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
            grad_input_padded = np.pad(grad_input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            X_padded = self.input
            grad_input_padded = grad_input
        
        # 计算梯度
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(new_H):
                    for w in range(new_W):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # 计算W的梯度
                        window = X_padded[b, :, h_start:h_end, w_start:w_end]
                        self.grads['W'][oc] += grads[b, oc, h, w] * window
                        
                        # 计算b的梯度
                        self.grads['b'][oc] += grads[b, oc, h, w]
                        
                        # 计算输入梯度
                        grad_input_padded[b, :, h_start:h_end, w_start:w_end] += grads[b, oc, h, w] * self.W[oc]
        
        # 如果有padding，需要去掉padding部分
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        # 如果有权重衰减
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.predicts = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
            
        # 计算交叉熵损失
        batch_size = predicts.shape[0]
        log_probs = -np.log(probs[range(batch_size), labels] + 1e-8)
        loss = np.sum(log_probs) / batch_size
        
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input

        batch_size = self.predicts.shape[0]
        
        if self.has_softmax:
            probs = softmax(self.predicts)
            probs[range(batch_size), self.labels] -= 1
            self.grads = probs / batch_size
        else:
            self.grads = np.zeros_like(self.predicts)
            self.grads[range(batch_size), self.labels] = -1.0 / (self.predicts[range(batch_size), self.labels] + 1e-8)
            self.grads /= batch_size
        
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition