import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Re-using previous Phish, DCWP, and LR definitions ---

class Phish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.gelu(x))

class DCWPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DCWPModule, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.phish = Phish()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.phish(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class LRModule(nn.Module):
    def __init__(self, channels):
        super(LRModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.phish1 = Phish()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.phish2 = Phish()
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.phish1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.phish2(out)
        out = self.bn2(out)
        out += residual
        return out

# --- NEW: CBAM Implementation ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 7x7 convolution is standard for the Spatial component of CBAM
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compress channels into 2: Average and Max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Sequential arrangement: Channel first, then Spatial
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# --- Updated LightMixer with CBAM ---

class LightMixerWithCBAM(nn.Module):
    def __init__(self, num_classes=10):
        super(LightMixerWithCBAM, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=3, padding=1, bias=False),
            Phish(),
            nn.BatchNorm2d(32)
        )
        
        self.layer1 = DCWPModule(32, 64)
        self.layer2 = DCWPModule(64, 128)
        self.layer3 = LRModule(128)
        self.layer4 = LRModule(128)
        
        # --- ADDED: CBAM Module ---
        # Input channels must match the output of layer4 (128)
        self.cbam = CBAM(in_planes=128, ratio=16)
        # --------------------------

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply Attention
        x = self.cbam(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Testing the implementation
if __name__ == "__main__":
    model = LightMixerWithCBAM(num_classes=10)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Output shape with CBAM: {output.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")