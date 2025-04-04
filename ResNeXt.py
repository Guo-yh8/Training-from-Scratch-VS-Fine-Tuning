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
device = torch.device("cpu")


# 数据加载
def load_data():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    return train_loader, test_loader


# ResNeXt模型结构
class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=8, base_width=4):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * cardinality
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += residual
        return F.relu(x)


class ResNeXtScratch(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(ResNeXtBlock(in_channels, out_channels, stride, cardinality=8))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 训练和验证函数
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


# 可视化函数
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
    plt.savefig('training_metrics.png')
    plt.show()


# 主函数
def main():
    total_start = time.time()

    train_loader, test_loader = load_data()
    model_scratch = ResNeXtScratch().to(device)
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.05, momentum=0.9)

    model_finetune = torchvision.models.resnext50_32x4d(
        weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    for param in model_finetune.parameters():
        param.requires_grad = False
    model_finetune.fc = nn.Linear(model_finetune.fc.in_features, 10)
    model_finetune = model_finetune.to(device)
    optimizer_finetune = optim.SGD(model_finetune.fc.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    epochs = 20
    history = {
        'scratch_train_loss': [], 'scratch_val_loss': [],
        'scratch_train_acc': [], 'scratch_val_acc': [],
        'finetune_train_loss': [], 'finetune_val_loss': [],
        'finetune_train_acc': [], 'finetune_val_acc': []
    }

    print("\n===== 开始训练 =====")
    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练和验证
        train_loss, train_acc = train_model(model_scratch, train_loader, criterion, optimizer_scratch, epoch)
        val_loss, val_acc = validate_model(model_scratch, test_loader, criterion)
        history['scratch_train_loss'].append(train_loss)
        history['scratch_val_loss'].append(val_loss)
        history['scratch_train_acc'].append(train_acc)
        history['scratch_val_acc'].append(val_acc)

        ft_train_loss, ft_train_acc = train_model(model_finetune, train_loader, criterion, optimizer_finetune, epoch)
        ft_val_loss, ft_val_acc = validate_model(model_finetune, test_loader, criterion)
        history['finetune_train_loss'].append(ft_train_loss)
        history['finetune_val_loss'].append(ft_val_loss)
        history['finetune_train_acc'].append(ft_train_acc)
        history['finetune_val_acc'].append(ft_val_acc)

        # 打印epoch结果
        epoch_time = time.time() - epoch_start
        print(f"\n[Epoch {epoch + 1}/{epochs}] 耗时: {epoch_time:.2f}秒")
        print(f"Scratch | Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        print(f"Finetune | Train Loss: {ft_train_loss:.4f} | Acc: {ft_train_acc:.1f}% | Val Acc: {ft_val_acc:.1f}%")
        print("=" * 60)

    # 训练后处理
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
