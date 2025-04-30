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
    def __init__(self, init_lr, model, weight_decay=0.0):  # 添加 weight_decay 参数
        super().__init__(init_lr, model)
        self.weight_decay = weight_decay
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                if hasattr(layer, 'grads') and layer.grads is not None:
                    for param_name in layer.params:
                        if param_name in layer.grads and layer.grads[param_name] is not None:
                            # 应用权重衰减
                            if self.weight_decay > 0:
                                layer.params[param_name] *= (1 - self.init_lr * self.weight_decay)
                            layer.params[param_name] -= self.init_lr * layer.grads[param_name]


class MomentGD(Optimizer):
    """带动量的梯度下降"""
    def __init__(self, init_lr, model, mu=0.9, lr_decay=1.0, mu_decay=1.0) -> None:
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        self.lr_decay = lr_decay  # 学习率衰减系数
        self.mu_decay = mu_decay  # 动量衰减系数
        self.velocity = {}  # 保存各参数的动量
        self.iterations = 0  # 迭代次数
        
        # 初始化动量
        for layer in self.model.layers:
            if layer.optimizable:
                for name, param in layer.params.items():
                    self.velocity[f"{id(layer)}.{name}"] = np.zeros_like(param)
    
    def step(self) -> None:
        self.iterations += 1
        current_lr = self.init_lr * (self.lr_decay ** self.iterations)
        current_mu = self.mu * (self.mu_decay ** self.iterations)
        
        for layer in self.model.layers:
            if layer.optimizable and hasattr(layer, 'grads'):
                for name, param in layer.params.items():
                    if name in layer.grads and layer.grads[name] is not None:
                        key = f"{id(layer)}.{name}"
                        # 更新动量
                        self.velocity[key] = current_mu * self.velocity[key] - current_lr * layer.grads[name]
                        # 更新参数
                        param += self.velocity[key]
        
        # 清空梯度
        self.zero_grad()