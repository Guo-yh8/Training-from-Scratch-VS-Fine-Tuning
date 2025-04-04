import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# 启用CPU多线程并设置随机种子
torch.set_num_threads(8)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据加载
def load_data():
    # Tiny ImageNet的标准化参数（使用ImageNet的默认参数）
    transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载Tiny ImageNet数据集
    train_dir = './data/tiny-imagenet-200/train' 
    val_dir = './data/tiny-imagenet-200/val'

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    return train_loader, test_loader


# ==================== DenseNet结构 ====================
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x))))


class DenseNetScratch(nn.Module):
    def __init__(self, num_classes=200, growth_rate=12, block_config=(3, 6, 12)):
        super().__init__()
        # 初始卷积
        self.features = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU()
        )

        # DenseBlock构建
        in_channels = 2 * growth_rate
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, in_channels, growth_rate)
            self.features.add_module(f'denseblock_{i + 1}', block)
            in_channels += num_layers * growth_rate
            if i != len(block_config) - 1:  # 最后一个block后不加Transition
                trans = TransitionLayer(in_channels, in_channels // 2)
                self.features.add_module(f'transition_{i + 1}', trans)
                in_channels = in_channels // 2

        # 分类头
        self.final_norm = nn.BatchNorm2d(in_channels)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(self.final_norm(features))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# ==================== 训练流程 ====================
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False) as progress:
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress.set_postfix({
                'Loss': f"{running_loss / (progress.n + 1):.3f}",
                'Acc': f"{100. * correct / total:.1f}%"
            })
    return running_loss / len(train_loader), 100. * correct / total


def validate_model(model, test_loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(test_loader), 100. * correct / total


def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history['scratch_train_loss'], 'b-', label='Scratch Train')
    plt.plot(history['scratch_val_loss'], 'b--', label='Scratch Val')
    plt.plot(history['finetune_train_loss'], 'r-', label='Finetune Train')
    plt.plot(history['finetune_val_loss'], 'r--', label='Finetune Val')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(122)
    plt.plot(history['scratch_train_acc'], 'b-', label='Scratch Train')
    plt.plot(history['scratch_val_acc'], 'b--', label='Scratch Val')
    plt.plot(history['finetune_train_acc'], 'r-', label='Finetune Train')
    plt.plot(history['finetune_val_acc'], 'r--', label='Finetune Val')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('densenet_metrics4.png')
    plt.show()


# ==================== 主函数 ====================
def main():
    total_start = time.time()  # 记录程序开始时间
    train_loader, test_loader = load_data()

    # 初始化模型
    model_scratch = DenseNetScratch().to(device)
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.05, momentum=0.9)

    # 微调预训练模型
    model_finetune = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    # 首先冻结所有参数
    for param in model_finetune.parameters():
        param.requires_grad = False
    # 解冻最后两个DenseBlock和分类头
    layers_to_unfreeze = [
        'features.denseblock3',  # 第三个DenseBlock
        'features.denseblock4',  # 第四个DenseBlock
        'features.norm5',  # 最后的BatchNorm
        'classifier'  # 分类层
    ]
    for name, param in model_finetune.named_parameters():
        if any(unfreeze_name in name for unfreeze_name in layers_to_unfreeze):
            param.requires_grad = True
    # 替换分类头
    model_finetune.classifier = nn.Linear(model_finetune.classifier.in_features, 200)
    model_finetune = model_finetune.to(device)
    # 优化器设置
    optimizer_finetune = optim.SGD([
        {'params': [p for n, p in model_finetune.named_parameters()
                    if 'denseblock3' in n or 'denseblock4' in n], 'lr': 0.001},  # 深层特征
        {'params': model_finetune.classifier.parameters(), 'lr': 0.01}  # 分类头
    ], momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    epochs = 50
    history = {
        'scratch_train_loss': [], 'scratch_val_loss': [],
        'scratch_train_acc': [], 'scratch_val_acc': [],
        'finetune_train_loss': [], 'finetune_val_loss': [],
        'finetune_train_acc': [], 'finetune_val_acc': []
    }

    print("\n===== 训练开始 =====")
    for epoch in range(epochs):
        epoch_start = time.time()

        # 从头训练
        train_loss, train_acc = train_model(model_scratch, train_loader, criterion, optimizer_scratch, epoch)
        val_loss, val_acc = validate_model(model_scratch, test_loader, criterion)
        history['scratch_train_loss'].append(train_loss)
        history['scratch_val_loss'].append(val_loss)
        history['scratch_train_acc'].append(train_acc)
        history['scratch_val_acc'].append(val_acc)

        # 微调训练
        ft_train_loss, ft_train_acc = train_model(model_finetune, train_loader, criterion, optimizer_finetune, epoch)
        ft_val_loss, ft_val_acc = validate_model(model_finetune, test_loader, criterion)
        history['finetune_train_loss'].append(ft_train_loss)
        history['finetune_val_loss'].append(ft_val_loss)
        history['finetune_train_acc'].append(ft_train_acc)
        history['finetune_val_acc'].append(ft_val_acc)

        # 打印结果
        epoch_time = time.time() - epoch_start
        print(f"\n[Epoch {epoch + 1}/{epochs}] 耗时: {epoch_time:.2f}秒")
        print(f"Scratch | Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        print(f"Finetune | Train Loss: {ft_train_loss:.4f} | Acc: {ft_train_acc:.1f}% | Val Acc: {ft_val_acc:.1f}%")
        print("=" * 60)

    plot_metrics(history)

    # 计算总时间
    total_time = time.time() - total_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\n===== 训练完成 =====\n"
          f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒\n"
          f"平均每个epoch耗时: {total_time / epochs:.2f}秒")


if __name__ == "__main__":
    main()
