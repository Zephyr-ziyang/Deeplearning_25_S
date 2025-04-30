from abc import abstractmethod
import numpy as np

class LRScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    def step(self) -> None: # 优化了一下学习率更新，设置重命名防止冲突
        self.step_count += 1
        self.update_lr()
    
    @abstractmethod
    def update_lr(self):
        pass

class StepLR(LRScheduler):
    """固定步长衰减学习率"""
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def update_lr(self):
        if self.step_count % self.step_size == 0:
            self.optimizer.init_lr *= self.gamma

class MultiStepLR(LRScheduler):
    """多阶段学习率衰减"""
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma
    
    def update_lr(self):
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(LRScheduler):
    """指数衰减学习率"""
    def __init__(self, optimizer, gamma=0.95) -> None:
        super().__init__(optimizer)
        self.gamma = gamma
    
    def update_lr(self):
        self.optimizer.init_lr *= self.gamma

class CosineAnnealingLR(LRScheduler):
    """余弦退火学习率, Deepseek-V3 认证(误)"""
    def __init__(self, optimizer, T_max, eta_min=0) -> None:
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.init_lr
    
    def update_lr(self):
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.step_count / self.T_max))
        self.optimizer.init_lr = lr