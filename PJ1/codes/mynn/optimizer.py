from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass
    def zero_grad(self):
        for layer in self.model.layers:
            if layer.optimizable == True and hasattr(layer, 'grads'):
                for key in layer.grads.keys():
                    if layer.grads[key] is not None:
                        layer.grads[key].fill(0)
    
class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        # 遍历所有可优化层
        for layer in self.model.layers:
            if layer.optimizable:
                # 确保梯度存在
                if hasattr(layer, 'grads') and layer.grads is not None:
                    for param_name in layer.params:
                        # 确保梯度和参数形状匹配
                        if param_name in layer.grads and layer.grads[param_name] is not None:
                            # 应用权重衰减（如果启用）
                            if layer.weight_decay:
                                layer.params[param_name] *= (1 - self.init_lr * layer.weight_decay_lambda)
                            # 更新参数
                            layer.params[param_name] -= self.init_lr * layer.grads[param_name]

    


class MomentGD(Optimizer):
    """带动量的梯度下降"""
    def __init__(self, init_lr, model, mu=0.9) -> None:
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        self.velocity = {}  # 保存各参数的动量
        
        # 初始化动量
        for name, param in self.model.params.items():
            self.velocity[name] = np.zeros_like(param)
    
    def step(self) -> None:
        for name, param in self.model.params.items():
            if name in self.model.grads:
                # 更新动量
                self.velocity[name] = self.mu * self.velocity[name] - self.init_lr * self.model.grads[name]
                # 更新参数
                param += self.velocity[name]
        
        # 清空梯度
        self.model.clear_grad()