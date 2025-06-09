import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from IPython import display
import time

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# 常量初始化
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128
seed_value = 42

# 设置路径
module_path = os.path.dirname(os.path.abspath(__file__))
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
print("home path:", home_path)

# 设置设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(f"cuda:{device_id[0]}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(device_id[0]))

# 设置随机种子
def set_random_seeds(seed_value=seed_value, device=device):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seeds()

# 加载数据
train_loader = get_cifar_loader(train=True, batch_size=batch_size)
val_loader = get_cifar_loader(train=False, batch_size=batch_size)

# 计算准确率
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

# 训练函数
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=50, model_name="vgg"):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_model_path = os.path.join(models_path, f'best_{model_name}.pth')
    
    for epoch in range(epochs_n):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs_n}')):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = get_accuracy(model, val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs_n} - '
              f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with val acc: {best_val_acc:.2f}%')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Loss Curve ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'Accuracy Curve ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(os.path.join(figures_path, f'training_curves_{model_name}.png'))
    plt.close()
    
    return train_losses, val_losses, train_accs, val_accs

# 损失景观分析
def analyze_loss_landscape(model, criterion, train_loader, model_name="vgg", num_steps=100):
    model.eval()
    min_curve = []
    max_curve = []
    
    # 获取一个batch的数据
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    # 计算原始损失和梯度
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # 保存原始参数
    original_params = [p.data.clone() for p in model.parameters()]
    
    # 定义学习率范围
    learning_rates = np.logspace(-5, -1, num=num_steps)
    
    for lr in learning_rates:
        # 临时修改参数
        with torch.no_grad():
            for param, orig in zip(model.parameters(), original_params):
                if param.grad is not None:
                    param.data = orig - lr * param.grad.data
        
        # 计算新损失
        with torch.no_grad():
            new_output = model(data)
            new_loss = criterion(new_output, target).item()
        
        min_curve.append(new_loss)
        max_curve.append(new_loss)
    
    # 恢复原始参数
    with torch.no_grad():
        for param, orig in zip(model.parameters(), original_params):
            param.data = orig
    
    # 绘制损失景观
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, min_curve, 'b-', label='Loss')
    plt.fill_between(learning_rates, min_curve, max_curve, alpha=0.3)
    plt.xscale('log')
    plt.title(f'Loss Landscape - {model_name}')
    plt.xlabel('Learning Rate (Step Size)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, f'loss_landscape_{model_name}.png'))
    plt.close()
    
    return min_curve, max_curve

# 主函数
def main():
    # 训练参数
    epochs = 50
    lr = 0.001
    
    # 训练无BN模型
    print("\nTraining VGG_A without BatchNorm...")
    set_random_seeds()
    model_no_bn = VGG_A()
    optimizer_no_bn = torch.optim.Adam(model_no_bn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses_no_bn, val_losses_no_bn, train_accs_no_bn, val_accs_no_bn = train(
        model_no_bn, optimizer_no_bn, criterion, train_loader, val_loader, 
        epochs_n=epochs, model_name="vgg_no_bn")
    
    # 损失景观分析（无BN）
    print("\nAnalyzing loss landscape for VGG_A without BatchNorm...")
    analyze_loss_landscape(model_no_bn, criterion, train_loader, model_name="vgg_no_bn")
    
    # 训练有BN模型
    print("\nTraining VGG_A with BatchNorm...")
    set_random_seeds()
    model_bn = VGG_A_BatchNorm()
    optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
    
    train_losses_bn, val_losses_bn, train_accs_bn, val_accs_bn = train(
        model_bn, optimizer_bn, criterion, train_loader, val_loader, 
        epochs_n=epochs, model_name="vgg_bn")
    
    # 损失景观分析（有BN）
    print("\nAnalyzing loss landscape for VGG_A with BatchNorm...")
    analyze_loss_landscape(model_bn, criterion, train_loader, model_name="vgg_bn")
    
    # 比较结果
    plt.figure(figsize=(15, 10))
    
    # 损失曲线比较
    plt.subplot(2, 2, 1)
    plt.plot(train_losses_no_bn, label='Without BN (Train)')
    plt.plot(val_losses_no_bn, label='Without BN (Val)')
    plt.plot(train_losses_bn, label='With BN (Train)')
    plt.plot(val_losses_bn, label='With BN (Val)')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线比较
    plt.subplot(2, 2, 2)
    plt.plot(train_accs_no_bn, label='Without BN (Train)')
    plt.plot(val_accs_no_bn, label='Without BN (Val)')
    plt.plot(train_accs_bn, label='With BN (Train)')
    plt.plot(val_accs_bn, label='With BN (Val)')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # 最终验证准确率比较
    plt.subplot(2, 2, 3)
    models = ['Without BN', 'With BN']
    final_accs = [val_accs_no_bn[-1], val_accs_bn[-1]]
    plt.bar(models, final_accs, color=['red', 'green'])
    plt.title('Final Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    
    # 训练速度比较
    plt.subplot(2, 2, 4)
    time_per_epoch = [sum(train_losses_no_bn)/len(train_losses_no_bn), 
                      sum(train_losses_bn)/len(train_losses_bn)]
    plt.bar(models, time_per_epoch, color=['red', 'green'])
    plt.title('Average Time per Epoch')
    plt.ylabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'bn_comparison_summary.png'))
    plt.close()
    
    print("\nExperiment completed successfully!")

if __name__ == '__main__':
    main()