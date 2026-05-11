# MODEL.py
import torch
import torch.nn as nn

class ImprovedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class MyFinalModel(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        # 第一层：快速降低空间分辨率 (112x112)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), # Conv3x3 stride=1 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.stage1 = nn.Sequential(
            ImprovedBlock(64, 128, stride = 2),
            ImprovedBlock(128, 128, stride = 1)
        )  
        self.stage2 = nn.Sequential(
            ImprovedBlock(128, 256, stride = 2),
            ImprovedBlock(256, 256, stride = 1)
        )
        self.stage3 = nn.Sequential(
            ImprovedBlock(256, 512, stride = 2),
            ImprovedBlock(512, 512, stride = 1)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        x = torch.cat([avg_x, max_x], dim=1)
        
        return self.fc(x)

def build_model(num_classes=37, device=None):
    model = MyFinalModel(num_classes=num_classes)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


