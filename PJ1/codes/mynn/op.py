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
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = initialize_method(scale=scale, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))  # 偏置通常初始化为0
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        
        # 计算fan_in
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        
        # 初始化权重 [out_channels, in_channels, kH, kW]
        self.W = initialize_method(scale=scale, size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = np.zeros((out_channels, 1))
        
        # 初始化梯度和参数
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
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
        new_H = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        new_W = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1
        
        # 使用im2col方法加速卷积运算
        X_col = im2col(X, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.out_channels, -1)
        output = np.dot(W_col, X_col) + self.b
        return output.reshape(self.out_channels, new_H, new_W, batch_size).transpose(3, 0, 1, 2)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, in_channels, H, W = self.input.shape
        out_channels, _, new_H, new_W = grads.shape
        
        # 1. 计算bias梯度
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        
        # 2. 使用im2col计算权重梯度
        X_col = im2col(self.input, self.kernel_size, self.stride, self.padding)
        grads_reshaped = grads.transpose(1, 2, 3, 0).reshape(out_channels, -1)
        self.grads['W'] = np.dot(grads_reshaped, X_col.T)
        self.grads['W'] = self.grads['W'].reshape(self.W.shape)
        
        # 3. 计算输入梯度
        W_reshaped = self.W.reshape(out_channels, -1)
        dX_col = np.dot(W_reshaped.T, grads_reshaped)
        grad_input = col2im(dX_col, self.input.shape, self.kernel_size, 
                           self.stride, self.padding)
        
        # 权重衰减
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
            
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

def col2im(col, input_shape, kernel_size, stride=1, padding=0):
    """将列矩阵转换回图像格式"""
    batch_size, in_channels, H, W = input_shape
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    
    out_H = (H + 2*padding - kH) // stride + 1
    out_W = (W + 2*padding - kW) // stride + 1
    
    # 初始化输出
    if padding > 0:
        img = np.zeros((batch_size, in_channels, H+2*padding, W+2*padding))
    else:
        img = np.zeros(input_shape)
    
    # 重建图像
    col_reshaped = col.reshape(in_channels*kH*kW, out_H*out_W*batch_size)
    for b in range(batch_size):
        for h in range(out_H):
            for w in range(out_W):
                h_start = h * stride
                w_start = w * stride
                h_end = h_start + kH
                w_end = w_start + kW
                
                img[b, :, h_start:h_end, w_start:w_end] += \
                    col_reshaped[:, b*out_H*out_W + h*out_W + w].reshape(in_channels, kH, kW)
    
    # 去除padding
    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img

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
        assert len(self.input.shape) == len(grads.shape), \
            f"Input dims {len(self.input.shape)} != Grads dims {len(grads.shape)}"
        
        # 梯度广播到输入形状
        if self.input.shape != grads.shape:
            grads = np.sum(grads, axis=tuple(range(len(grads.shape) - len(self.input.shape))), keepdims=True)
            grads = np.broadcast_to(grads, self.input.shape)
            
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
            
        # 建议增加数值稳定性处理
        probs = np.clip(probs, 1e-10, 1.0)
        log_probs = -np.log(probs[range(len(labels)), labels])
        loss = np.mean(log_probs)
        
        return loss
    
    def backward(self):
        batch_size = self.predicts.shape[0]
        
        if self.has_softmax:
            # 计算softmax梯度
            probs = softmax(self.predicts)
            # 正确计算梯度：∂L/∂z = p - y
            grad = probs.copy()
            grad[range(batch_size), self.labels] -= 1
            grad /= batch_size  # 平均梯度
        else:
            # 直接计算交叉熵梯度
            grad = np.zeros_like(self.predicts)
            grad[range(batch_size), self.labels] = -1.0 / (self.predicts[range(batch_size), self.labels] + 1e-8)
            grad /= batch_size
        
        # 确保调用模型的backward方法
        if self.model is not None:
            self.model.backward(grad)

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
    partition = np.clip(partition, 1e-10, np.inf)
    return x_exp / partition


def im2col(X, kernel_size, stride=1, padding=0):
    """
    将输入图像转换为列矩阵以便高效卷积计算
    参数:
        X: 输入数据 [batch_size, in_channels, H, W]
        kernel_size: 卷积核大小 (kH, kW)
        stride: 步长
        padding: 填充大小
    返回:
        col: 2D矩阵 [kH*kW*in_channels, out_H*out_W*batch_size]
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
        
    batch_size, in_channels, H, W = X.shape
    kH, kW = kernel_size
    
    # 计算输出尺寸
    out_H = (H + 2*padding - kH) // stride + 1
    out_W = (W + 2*padding - kW) // stride + 1
    
    # 添加padding
    if padding > 0:
        X_padded = np.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), 
                         mode='constant')
    else:
        X_padded = X
    
    # 初始化输出矩阵
    col = np.zeros((kH*kW*in_channels, out_H*out_W*batch_size))
    
    # 填充输出矩阵
    for b in range(batch_size):
        for h in range(out_H):
            for w in range(out_W):
                h_start = h * stride
                w_start = w * stride
                h_end = h_start + kH
                w_end = w_start + kW
                
                # 提取当前窗口并展平
                window = X_padded[b, :, h_start:h_end, w_start:w_end]
                col[:, b*out_H*out_W + h*out_W + w] = window.reshape(-1)
    
    return col