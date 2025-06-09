import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
import os
import time
import json
import copy
import math

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建输出目录
output_dir = os.path.join(script_dir, "TOPIC_CIFAR10")
os.makedirs(output_dir, exist_ok=True)
print(f"所有输出文件将保存到: {output_dir}")

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

# 定义类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 标签平滑交叉熵损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, targets)
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes

# 计算类别权重
def get_class_weights():
    class_counts = np.zeros(10)
    for _, target in train_loader:
        for t in target.numpy():
            class_counts[t] += 1
    weights = 1.0 / class_counts
    weights /= weights.sum()
    return torch.FloatTensor(weights).to(device)

# 定义高性能CNN模型（支持多种配置）
class HighPerfCIFAR10Model(nn.Module):
    def __init__(self, num_classes=10, channels=[64, 128, 256], activation=nn.LeakyReLU(0.1)):
        super(HighPerfCIFAR10Model, self).__init__()
        self.activation = activation
        
        # 初始卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            activation,
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # 残差块1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            activation,
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            activation,
        )
        
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[1])
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # 残差块2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            activation,
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            activation,
        )
        
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[2])
        )
        
        # 最终分类层
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[2], 512),
            nn.BatchNorm1d(512),
            activation,
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        
        # 第一个残差连接
        identity = self.shortcut1(x)
        x = self.res_block1(x)
        x = x + identity
        x = self.activation(x)
        
        x = self.conv2(x)
        
        # 第二个残差连接
        identity = self.shortcut2(x)
        x = self.res_block2(x)
        x = x + identity
        x = self.activation(x)
        
        x = self.fc(x)
        return x

# 训练模型函数
def train_model(model, optimizer, criterion, epochs=100, scheduler=None, experiment_name="base"):
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    best_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    all_losses = []  # 用于损失景观分析
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            all_losses.append(loss.item())
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'[{experiment_name}] Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        print(f'[{experiment_name}] Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({train_acc:.2f}%)')
        
        # 测试
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        print(f'[{experiment_name}] Test set: Average loss: {test_loss:.4f}, Accuracy: {test_correct}/{test_total} ({test_acc:.2f}%)')
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_{experiment_name}.pth'))
            print(f'[{experiment_name}] New best model saved with accuracy: {best_acc:.2f}%')
        
        # 每10个epoch可视化一次
        if epoch == 1 or epoch % 10 == 0:
            visualize_filters(model, epoch, experiment_name)
            plot_confusion_matrix(all_targets, all_predictions, classes, epoch, experiment_name)
            
            # 绘制训练曲线
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.title(f'Loss Curve ({experiment_name})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(test_accs, label='Test Accuracy')
            plt.title(f'Accuracy Curve ({experiment_name})')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'training_curves_{experiment_name}.png'))
            plt.close()
            
            # 可视化特征图
            sample_img, _ = next(iter(test_loader))
            visualize_feature_maps(model, sample_img[0], epoch, experiment_name)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 最终可视化
    visualize_filters(model, "final", experiment_name)
    plot_confusion_matrix(all_targets, all_predictions, classes, "final", experiment_name)
    
    # 返回训练结果
    results = {
        'best_test_acc': best_acc,
        'training_time': training_time,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'all_losses': all_losses,
        'params': sum(p.numel() for p in model.parameters())
    }
    
    return results

# 可视化滤波器
def visualize_filters(model, epoch, experiment_name, layer_idx=0):
    weights = model.conv1[layer_idx].weight.data.cpu().numpy()
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle(f'Conv Filters - {experiment_name} (Epoch {epoch})', fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            # 对于RGB图像，取第一个通道可视化
            filter_img = weights[i, 0]  # 只取第一个通道
            # 归一化
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
            ax.imshow(filter_img, cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'conv_filters_{experiment_name}_epoch_{epoch}.png'))
    plt.close()

# 混淆矩阵绘制函数
def plot_confusion_matrix(targets, predictions, classes, epoch, experiment_name):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {experiment_name} (Epoch {epoch})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{experiment_name}_epoch_{epoch}.png'))
    plt.close()

# 可视化特征图
def visualize_feature_maps(model, sample_img, epoch, experiment_name):
    model.eval()
    activations = {}
    
    # 注册钩子
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            def hook(module, input, output, name=name):
                activations[name] = output.detach()
            hooks.append(layer.register_forward_hook(hook))
    
    # 前向传播
    with torch.no_grad():
        model(sample_img.unsqueeze(0).to(device))
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化特征图
    for layer_name, feature_maps in activations.items():
        if len(feature_maps.shape) < 4:
            continue
            
        maps = feature_maps[0].cpu()
        n_maps = min(maps.size(0), 16)  # 最多显示16个
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        fig.suptitle(f'Feature Maps: {layer_name} - {experiment_name} (Epoch {epoch})', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < n_maps:
                if len(maps[i].shape) == 2:  # 单通道特征图
                    ax.imshow(maps[i], cmap='viridis')
                else:  # 多通道特征图，取平均值
                    ax.imshow(maps[i].mean(0), cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_maps_{experiment_name}_{layer_name}_epoch_{epoch}.png'))
        plt.close()

# 损失景观可视化
def visualize_loss_landscape(model, criterion, train_loader, experiment_name="loss_landscape"):
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    max_curve = []
    min_curve = []
    
    for lr in learning_rates:
        print(f"Training with learning rate: {lr} for loss landscape analysis")
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        optimizer = optim.SGD(model_copy.parameters(), lr=lr)
        
        losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 50:  # 只使用部分数据加快计算
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_copy(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        max_curve.append(np.max(losses))
        min_curve.append(np.min(losses))
        print(f"LR: {lr:.0e}, Max Loss: {np.max(losses):.4f}, Min Loss: {np.min(losses):.4f}")
    
    # 绘制损失景观
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, max_curve, 'r-', label='Max Loss')
    plt.plot(learning_rates, min_curve, 'b-', label='Min Loss')
    plt.fill_between(learning_rates, min_curve, max_curve, color='gray', alpha=0.3)
    plt.xscale('log')
    plt.title(f'Loss Landscape Analysis - {experiment_name}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_landscape_{experiment_name}.png'))
    plt.close()
    
    return max_curve, min_curve

# 生成性能报告
def generate_performance_report(model, test_acc, train_time, experiment_name):
    report = {
        "experiment_name": experiment_name,
        "test_accuracy": test_acc,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_time": measure_inference_time(model),
        "training_time": train_time,
        "model_structure": str(model).replace('\n', ' ')
    }
    
    # 保存报告
    with open(os.path.join(output_dir, f'performance_report_{experiment_name}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# 测量推理时间
def measure_inference_time(model, n_runs=100):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # 预热
    for _ in range(10):
        _ = model(dummy_input)
    
    # 计时
    start = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input)
    end = time.time()
    
    return (end - start) / n_runs * 1000  # 毫秒/样本

# 绘制优化器比较
def plot_optimizer_comparison(results):
    names = [r['optimizer'] for r in results]
    accs = [r['test_acc'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, accs, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Optimizer Performance Comparison')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(min(accs)-5, max(accs)+5)
    plt.savefig(os.path.join(output_dir, 'optimizer_comparison.png'))
    plt.close()

# 绘制类别准确率
def plot_class_accuracy(model, test_loader, classes, experiment_name):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            c = (predicted == target)
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, [class_correct[i]/class_total[i] for i in range(10)])
    plt.title(f'Accuracy per Class - {experiment_name}')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_accuracy_{experiment_name}.png'))
    plt.close()
    
    return [class_correct[i]/class_total[i] for i in range(10)]

# 运行滤波器数量实验
def run_filter_experiments():
    filter_configs = [
        {'name': 'small', 'channels': [32, 64, 128]},
        {'name': 'medium', 'channels': [64, 128, 256]},  # 原始配置
        {'name': 'large', 'channels': [128, 256, 512]}
    ]
    
    results = []
    for config in filter_configs:
        print(f"\n=== Running filter experiment: {config['name']} ===")
        model = HighPerfCIFAR10Model(channels=config['channels'])
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        exp_name = f"filters_{config['name']}"
        result = train_model(model, optimizer, criterion, epochs=50, experiment_name=exp_name)
        
        # 保存结果
        report = generate_performance_report(
            model, result['best_test_acc'], 
            result['training_time'], exp_name)
        
        # 绘制类别准确率
        class_acc = plot_class_accuracy(model, test_loader, classes, exp_name)
        
        results.append({
            'config': config,
            'test_acc': result['best_test_acc'],
            'params': report['total_parameters'],
            'inference_time': report['inference_time'],
            'class_acc': class_acc
        })
    
    return results

# 运行损失函数实验
def run_loss_experiments():
    loss_functions = [
        {'name': 'CrossEntropy', 'fn': nn.CrossEntropyLoss()},
        {'name': 'WeightedCrossEntropy', 'fn': nn.CrossEntropyLoss(weight=get_class_weights())},
        {'name': 'LabelSmoothing', 'fn': LabelSmoothingCrossEntropy(smoothing=0.1)}
    ]
    
    results = []
    for loss in loss_functions:
        print(f"\n=== Running loss function experiment: {loss['name']} ===")
        model = HighPerfCIFAR10Model()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        exp_name = f"loss_{loss['name']}"
        result = train_model(model, optimizer, loss['fn'], epochs=50, experiment_name=exp_name)
        
        # 保存结果
        report = generate_performance_report(
            model, result['best_test_acc'], 
            result['training_time'], exp_name)
        
        # 绘制类别准确率
        class_acc = plot_class_accuracy(model, test_loader, classes, exp_name)
        
        results.append({
            'loss': loss['name'],
            'test_acc': result['best_test_acc'],
            'params': report['total_parameters'],
            'inference_time': report['inference_time'],
            'class_acc': class_acc
        })
    
    return results

# 运行激活函数实验
def run_activation_experiments():
    activations = [
        {'name': 'LeakyReLU', 'fn': nn.LeakyReLU(0.1)},
        {'name': 'ReLU', 'fn': nn.ReLU()},
        {'name': 'ELU', 'fn': nn.ELU()},
        {'name': 'GELU', 'fn': nn.GELU()}
    ]
    
    results = []
    for act in activations:
        print(f"\n=== Running activation function experiment: {act['name']} ===")
        model = HighPerfCIFAR10Model(activation=act['fn'])
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        exp_name = f"activation_{act['name']}"
        result = train_model(model, optimizer, criterion, epochs=50, experiment_name=exp_name)
        
        # 保存结果
        report = generate_performance_report(
            model, result['best_test_acc'], 
            result['training_time'], exp_name)
        
        # 绘制类别准确率
        class_acc = plot_class_accuracy(model, test_loader, classes, exp_name)
        
        results.append({
            'activation': act['name'],
            'test_acc': result['best_test_acc'],
            'params': report['total_parameters'],
            'inference_time': report['inference_time'],
            'class_acc': class_acc
        })
    
    return results

# 运行优化器实验
def run_optimizer_experiments():
    optimizers = [
        {'name': 'AdamW', 'fn': lambda m: optim.AdamW(m.parameters(), lr=0.001)},
        {'name': 'SGD', 'fn': lambda m: optim.SGD(m.parameters(), lr=0.01, momentum=0.9)},
        {'name': 'RMSprop', 'fn': lambda m: optim.RMSprop(m.parameters(), lr=0.001)},
        {'name': 'Adagrad', 'fn': lambda m: optim.Adagrad(m.parameters(), lr=0.01)},
        {'name': 'Adam', 'fn': lambda m: optim.Adam(m.parameters(), lr=0.001)}
    ]
    
    results = []
    for opt in optimizers:
        print(f"\n=== Running optimizer experiment: {opt['name']} ===")
        model = HighPerfCIFAR10Model().to(device)
        optimizer = opt['fn'](model)
        criterion = nn.CrossEntropyLoss()
        exp_name = f"optimizer_{opt['name']}"
        result = train_model(model, optimizer, criterion, epochs=50, experiment_name=exp_name)
        
        # 保存结果
        report = generate_performance_report(
            model, result['best_test_acc'], 
            result['training_time'], exp_name)
        
        # 绘制类别准确率
        class_acc = plot_class_accuracy(model, test_loader, classes, exp_name)
        
        results.append({
            'optimizer': opt['name'],
            'test_acc': result['best_test_acc'],
            'params': report['total_parameters'],
            'inference_time': report['inference_time'],
            'class_acc': class_acc
        })
    
    # 绘制优化器比较图
    plot_optimizer_comparison(results)
    
    return results

# 生成综合报告
def generate_comprehensive_report(base_results, filter_results, loss_results, activation_results, optimizer_results):
    report = {
        "base_model": {
            "test_accuracy": base_results['best_test_acc'],
            "params": base_results['params'],
            "training_time": base_results['training_time']
        },
        "filter_experiments": filter_results,
        "loss_experiments": loss_results,
        "activation_experiments": activation_results,
        "optimizer_experiments": optimizer_results
    }
    
    # 保存报告
    with open(os.path.join(output_dir, 'comprehensive_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # 可视化比较结果
    plot_comparison_charts(report)
    
    return report

# 绘制比较图表
def plot_comparison_charts(report):
    # 滤波器配置比较
    filter_names = [f"{f['config']['name']} ({f['params']//1000}k)" for f in report['filter_experiments']]
    filter_accs = [f['test_acc'] for f in report['filter_experiments']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(filter_names, filter_accs)
    plt.title('Filter Configuration Comparison')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(min(filter_accs)-5, max(filter_accs)+5)
    plt.savefig(os.path.join(output_dir, 'filter_config_comparison.png'))
    plt.close()
    
    # 损失函数比较
    loss_names = [l['loss'] for l in report['loss_experiments']]
    loss_accs = [l['test_acc'] for l in report['loss_experiments']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(loss_names, loss_accs)
    plt.title('Loss Function Comparison')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(min(loss_accs)-5, max(loss_accs)+5)
    plt.savefig(os.path.join(output_dir, 'loss_function_comparison.png'))
    plt.close()
    
    # 激活函数比较
    act_names = [a['activation'] for a in report['activation_experiments']]
    act_accs = [a['test_acc'] for a in report['activation_experiments']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(act_names, act_accs)
    plt.title('Activation Function Comparison')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(min(act_accs)-5, max(act_accs)+5)
    plt.savefig(os.path.join(output_dir, 'activation_comparison.png'))
    plt.close()
    
    # 优化器比较（已有单独函数）

# # 主函数
# def main():
#     # 基础模型训练
#     print("\n=== Training Base Model ===")
#     base_model = HighPerfCIFAR10Model()
#     base_optimizer = optim.AdamW(base_model.parameters(), lr=0.001)
#     base_criterion = nn.CrossEntropyLoss()
#     base_results = train_model(
#         base_model, base_optimizer, base_criterion, 
#         epochs=100, experiment_name="base_model")
    
#     # 生成基础模型性能报告
#     base_report = generate_performance_report(
#         base_model, base_results['best_test_acc'], 
#         base_results['training_time'], "base_model")
    
#     # 绘制基础模型类别准确率
#     base_class_acc = plot_class_accuracy(
#         base_model, test_loader, classes, "base_model")
    
#     # 损失景观分析
#     visualize_loss_landscape(
#         base_model, base_criterion, train_loader, 
#         experiment_name="base_model")
    
#     # 运行对比实验
#     print("\n=== Running Filter Experiments ===")
#     filter_results = run_filter_experiments()
    
#     print("\n=== Running Loss Function Experiments ===")
#     loss_results = run_loss_experiments()
    
#     print("\n=== Running Activation Function Experiments ===")
#     activation_results = run_activation_experiments()
    
#     print("\n=== Running Optimizer Experiments ===")
#     optimizer_results = run_optimizer_experiments()
    
#     # 生成综合报告
#     comprehensive_report = generate_comprehensive_report(
#         base_results, filter_results, loss_results, 
#         activation_results, optimizer_results)
    
#     print("\n=== All Experiments Completed ===")
#     print(f"Base model accuracy: {base_results['best_test_acc']:.2f}%")
#     print(f"Best overall accuracy: {max([r['test_acc'] for r in optimizer_results]):.2f}%")
#     print(f"All results saved to: {output_dir}")

def main():
    # 注释掉已完成的部分
    # print("\n=== Training Base Model ===")
    # base_model = HighPerfCIFAR10Model()
    # ... (基础模型训练代码)
    
    # 注释掉已完成的部分
    # print("\n=== Running Filter Experiments ===")
    # filter_results = run_filter_experiments()
    
    # 注释掉已完成的部分
    # print("\n=== Running Loss Function Experiments ===")
    # loss_results = run_loss_experiments()
    
    # 注释掉已完成的部分
    # print("\n=== Running Activation Function Experiments ===")
    # activation_results = run_activation_experiments()
    
    # 运行优化器实验
    print("\n=== Running Optimizer Experiments ===")
    optimizer_results = run_optimizer_experiments()
    
    # 由于我们跳过了前面的实验，需要手动创建一些结果占位符
    base_results = {
        'best_test_acc': 92.0,  # 替换为您实际的基础模型准确率
        'params': 1327050,      # 替换为您实际的基础模型参数数量
        'training_time': 0       # 替换为实际训练时间
    }
    
    filter_results = [{
        "experiment_name": "filters_large",
        "test_accuracy": 92.5,
        "total_parameters": 5014922,
        "trainable_parameters": 5014922,
        "inference_time": 1.0619854927062988,
        "training_time": 563.9855027198792,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=512, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "filters_medium",
        "test_accuracy": 90.03,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0684609413146973,
        "training_time": 470.1682379245758,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "filters_small",
        "test_accuracy": 88.48,
        "total_parameters": 370922,
        "trainable_parameters": 370922,
        "inference_time": 1.0358309745788574,
        "training_time": 468.950630903244,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=128, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        }
    ]  
    loss_results = [
        {
        "experiment_name": "loss_CrossEntropy",
        "test_accuracy": 90.25,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0431551933288574,
        "training_time": 470.8683006763458,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "loss_LabelSmoothing",
        "test_accuracy": 90.6,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0461759567260742,
        "training_time": 469.2769377231598,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "loss_WeightedCrossEntropy",
        "test_accuracy": 91.0,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0880756378173828,
        "training_time": 472.9771604537964,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        }
    ]    # 替换为您实际的损失函数实验结果
    
    activation_results = [
        {
        "experiment_name": "activation_ELU",
        "test_accuracy": 89.67,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0741066932678223,
        "training_time": 473.90342926979065,
        "model_structure": "HighPerfCIFAR10Model(   (activation): ELU(alpha=1.0)   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): ELU(alpha=1.0)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): ELU(alpha=1.0)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): ELU(alpha=1.0)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): ELU(alpha=1.0)   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): ELU(alpha=1.0)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): ELU(alpha=1.0)   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): ELU(alpha=1.0)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "activation_GELU",
        "test_accuracy": 91.44,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.094191074371338,
        "training_time": 463.6625621318817,
        "model_structure": "HighPerfCIFAR10Model(   (activation): GELU(approximate='none')   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): GELU(approximate='none')     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): GELU(approximate='none')     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): GELU(approximate='none')     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): GELU(approximate='none')   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): GELU(approximate='none')     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): GELU(approximate='none')   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): GELU(approximate='none')     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "activation_LeakyReLU",
        "test_accuracy": 91.67,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0669612884521484,
        "training_time": 471.85235142707825,
        "model_structure": "HighPerfCIFAR10Model(   (activation): LeakyReLU(negative_slope=0.1)   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): LeakyReLU(negative_slope=0.1)     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): LeakyReLU(negative_slope=0.1)   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): LeakyReLU(negative_slope=0.1)     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        },
        {
        "experiment_name": "activation_ReLU",
        "test_accuracy": 91.41,
        "total_parameters": 1327050,
        "trainable_parameters": 1327050,
        "inference_time": 1.0673856735229492,
        "training_time": 460.9864625930786,
        "model_structure": "HighPerfCIFAR10Model(   (activation): ReLU()   (conv1): Sequential(     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): ReLU()     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): ReLU()     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (7): Dropout(p=0.2, inplace=False)   )   (res_block1): Sequential(     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): ReLU()     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): ReLU()   )   (shortcut1): Sequential(     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (conv2): Sequential(     (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     (1): Dropout(p=0.3, inplace=False)   )   (res_block2): Sequential(     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (2): ReLU()     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (5): ReLU()   )   (shortcut2): Sequential(     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)   )   (fc): Sequential(     (0): AdaptiveAvgPool2d(output_size=(1, 1))     (1): Flatten(start_dim=1, end_dim=-1)     (2): Linear(in_features=256, out_features=512, bias=True)     (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     (4): ReLU()     (5): Dropout(p=0.5, inplace=False)     (6): Linear(in_features=512, out_features=10, bias=True)   ) )"
        }
    ]


    # 生成综合报告
    comprehensive_report = generate_comprehensive_report(
        base_results, filter_results, loss_results, 
        activation_results, optimizer_results)
    
    print("\n=== All Experiments Completed ===")
    print(f"Base model accuracy: {base_results['best_test_acc']:.2f}%")
    print(f"Best overall accuracy: {max([r['test_acc'] for r in optimizer_results]):.2f}%")
    print(f"All results saved to: {output_dir}")

if __name__ == '__main__':
    main()