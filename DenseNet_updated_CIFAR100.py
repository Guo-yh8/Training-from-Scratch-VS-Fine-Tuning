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

torch.set_num_threads(8)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据加载
def load_data():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    return train_loader, test_loader


# ==================== 升级版DenseNet结构 ====================
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.2):
        super().__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate=0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class DenseNetScratch(nn.Module):
    def __init__(self, num_classes=100, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16]):
        super().__init__()
        bn_size = 4
        drop_rate = 0.2

        # 初始卷积层
        self.conv1 = Conv1(in_planes=3, places=init_channels)

        # DenseBlock构建
        num_features = init_channels
        self.layer1 = DenseBlock(num_layers=blocks[0], in_channels=num_features,
                                 growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(num_features, num_features // 2)
        num_features = num_features // 2

        self.layer2 = DenseBlock(num_layers=blocks[1], in_channels=num_features,
                                 growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(num_features, num_features // 2)
        num_features = num_features // 2

        self.layer3 = DenseBlock(num_layers=blocks[2], in_channels=num_features,
                                 growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(num_features, num_features // 2)
        num_features = num_features // 2

        self.layer4 = DenseBlock(num_layers=blocks[3], in_channels=num_features,
                                 growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features += blocks[3] * growth_rate

        # 分类头
        self.final_norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)

        x = F.relu(self.final_norm(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
    plt.savefig('densenet_metrics2.png')
    plt.show()


# ==================== 主函数 ====================
def main():
    total_start = time.time()
    train_loader, test_loader = load_data()

    # 初始化模型 - 使用更深的DenseNet结构
    model_scratch = DenseNetScratch(num_classes=100, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16]).to(device)
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    # 微调预训练模型
    model_finetune = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    for param in model_finetune.parameters():
        param.requires_grad = False

    # 解冻最后两个DenseBlock和分类头
    layers_to_unfreeze = ['features.denseblock3', 'features.denseblock4', 'features.norm5', 'classifier']
    for name, param in model_finetune.named_parameters():
        if any(unfreeze_name in name for unfreeze_name in layers_to_unfreeze):
            param.requires_grad = True

    model_finetune.classifier = nn.Linear(model_finetune.classifier.in_features, 100)
    model_finetune = model_finetune.to(device)

    optimizer_finetune = optim.SGD([
        {'params': [p for n, p in model_finetune.named_parameters() if 'denseblock3' in n or 'denseblock4' in n],
         'lr': 0.001},
        {'params': model_finetune.classifier.parameters(), 'lr': 0.01}
    ], momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    epochs = 100
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

        epoch_time = time.time() - epoch_start
        print(f"\n[Epoch {epoch + 1}/{epochs}] 耗时: {epoch_time:.2f}秒")
        print(f"Scratch | Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        print(f"Finetune | Train Loss: {ft_train_loss:.4f} | Acc: {ft_train_acc:.1f}% | Val Acc: {ft_val_acc:.1f}%")
        print("=" * 60)

    plot_metrics(history)

    total_time = time.time() - total_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\n===== 训练完成 =====\n"
          f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒\n"
          f"平均每个epoch耗时: {total_time / epochs:.2f}秒")


if __name__ == "__main__":
    main()
