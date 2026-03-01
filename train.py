# 导入必要库
import random
import numpy as np
import torch
import torch.nn as tn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ===================== 1. 固定随机种子（保证结果可复现） =====================
def set_seed(seed=30):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(30)

# ===================== 2. 定义模型（仅模型结构，无其他逻辑） =====================
class PetClassifier(tn.Module):
    def __init__(self, num_classes=37):
        super(PetClassifier, self).__init__()

        # 特征提取：5层卷积+池化（224→7）
        self.features = tn.Sequential(
            tn.Conv2d(3, 32, kernel_size=3, padding=1),
            tn.ReLU(inplace=True),
            tn.MaxPool2d(kernel_size=2, stride=2),

            tn.Conv2d(32, 64, kernel_size=3, padding=1),
            tn.ReLU(inplace=True),
            tn.MaxPool2d(kernel_size=2, stride=2),

            tn.Conv2d(64, 128, kernel_size=3, padding=1),
            tn.ReLU(inplace=True),
            tn.MaxPool2d(kernel_size=2, stride=2),

            tn.Conv2d(128, 256, kernel_size=3, padding=1),
            tn.ReLU(inplace=True),
            tn.MaxPool2d(kernel_size=2, stride=2),

            tn.Conv2d(256, 512, kernel_size=3, padding=1),
            tn.ReLU(inplace=True),
            tn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器：全连接层
        self.classifier = tn.Sequential(
            tn.Dropout(p=0.5),
            tn.Linear(512 * 7 * 7, 512),
            tn.ReLU(inplace=True),
            tn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ===================== 3. 数据加载（适配Windows系统） =====================
# 训练集增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试/验证集仅标准化
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集（下载完成后会缓存，下次运行不重复下载）
trainval_aug = datasets.OxfordIIITPet(root="./data", split="trainval", transform=train_transform, download=True)
trainval_noaug = datasets.OxfordIIITPet(root="./data", split="trainval", transform=test_transform, download=True)
test_dataset = datasets.OxfordIIITPet(root="./data", split="test", transform=test_transform, download=True)

# 拆分训练/验证集
train_size = int(0.9 * len(trainval_aug))
val_size = len(trainval_aug) - train_size
train_dataset, val_indices = random_split(trainval_aug, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_dataset = torch.utils.data.Subset(trainval_noaug, val_indices.indices)

# 数据加载器（num_workers=0适配Windows）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# ===================== 4. 训练+测试主逻辑 =====================
if __name__ == "__main__":
    # 设备选择（自动识别GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")

    # 初始化模型
    model = PetClassifier(num_classes=37).to(device)
    criterion = tn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30 

    # 训练循环
    print("开始训练（共30个epoch）...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播+更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 打印日志
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_val_loss:.4f} | 验证精度：{val_acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), "pet_classifier.pth")
    print("\n训练完成！模型已保存为 pet_classifier.pth")

    # 测试模型
    print("开始测试模型...")
    model.load_state_dict(torch.load("pet_classifier.pth", map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"测试集最终精度：{test_acc:.2f}%")