# MNIST手写数字分类神经网络项目

本项目基于NumPy实现了神经网络及其变体，用于MNIST手写数字分类。目标是通过调整网络结构、优化策略、正则化方法以及实现卷积神经网络（CNN）来提升模型性能。

---

## 📌 环境要求
- Python 3.8+
- NumPy
- Matplotlib（可视化工具）
- tqdm（可选，用于进度条）


# 问题解决

1. 支持自由的层数设置：
   ```python
        if size_list is not None and act_func is not None:
            self.layers = []
            # 更多层数选择
            for i in range(len(size_list)-1):
                # 添加线性层
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i+1])
                if lambda_list and i < len(lambda_list):
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                
                # 激活函数添加
                if i < len(size_list)-2 and act_func == 'ReLU':
                    self.layers.append(ReLU())
   ```
