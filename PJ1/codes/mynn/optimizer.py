from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


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