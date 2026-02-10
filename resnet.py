"""
In this file we implement the ResNet architecture described in https://arxiv.org/pdf/1512.03385

Layers:
Conv Input (stride 2, 64 channels) - Halves the image (224 -> 112)
Max Pool (stride 2) - Halves the image (112 -> 56)
Stack 1 (no downsampling, 64 channels) - Image size unchanged (56 -> 56)
Stack 2 (downsampling at start, 128 channels) - Halves the image (56 -> 28)
Stack 3 (downsampling at start, 256 channels) - Halves the image (28 -> 14)
Stack 4 (downsampling at start, 512 channels) - Halves the image (14 -> 7)
Average Pool - Reduces each feature map to a single number (7 -> 1)
Fully Connected Output (n classes) - Reduces to n classes (512 -> n)
Softmax - Converts to n probabilities

The paper describes this architecture and the convolutions sizes for ImageNet.

The paper also mentions training in CIFAR-10, and gives a simpler architecture, with only 3 stacks,
with 2 blocks per stack, and channels 16, 32, and 64. It also changes the initial convolution
to 3x3 with stride 1 and padding 1, and removes the max pool. This results in the original image
of size 32 to end with an image of size 8 after the 3 stacks, similar to the 7x7 output of the ImageNet architecture.

Our objective is to train the ResNet-18, described for ImageNet, on CIFAR-100.
If we were to train with the exact same architecture, the initial image of size 32 would
loose a lot of information in the first few layers, and the final image size after the
4 stacks would be already 1x1, rendering the average pool useless.

So we will have to make some compromises for better performance; notably in the first convolution and max pool layers,
as we don't want to to loose too much information early on by halving the image size too many times.

Here we will describe 2 different architectures:
  - one more similar to the ImageNet architecture,
  - and one more similar to the CIFAR-10 architecture.

We will train both and compare results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block described for ResNet-18 and ResNet-34 in the paper

    Also used for ResNet Cifar-10

    Basically 2 3x3 convolutions with a skip connection, and an optional downsampling in the first convolution if stride > 1
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.first_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.second_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.first_conv(x)))
        out = self.bn2(self.second_conv(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_ImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stack1 = self._make_stack(block, 64, num_blocks[0], first_block_stride=1)
        self.stack2 = self._make_stack(block, 128, num_blocks[1], first_block_stride=2)
        self.stack3 = self._make_stack(block, 256, num_blocks[2], first_block_stride=2)
        self.stack4 = self._make_stack(block, 512, num_blocks[3], first_block_stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_stack(self, block, out_channels, num_blocks, first_block_stride):
        strides = [first_block_stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.stack4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.softmax(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stack1 = self._make_stack(block, 16, num_blocks[0], first_block_stride=1)
        self.stack2 = self._make_stack(block, 32, num_blocks[1], first_block_stride=2)
        self.stack3 = self._make_stack(block, 64, num_blocks[2], first_block_stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_stack(self, block, out_channels, num_blocks, first_block_stride):
        strides = [first_block_stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    block = ResidualBlock(3, 64, stride=2)
    out = block(x)
    assert out.shape == (
        1,
        64,
        16,
        16,
    ), f"Expected output shape (1, 64, 16, 16), but got {out.shape}"

    x = torch.randn(1, 3, 32, 32)
    block = ResidualBlock(3, 64, stride=1)
    out = block(x)
    assert out.shape == (
        1,
        64,
        32,
        32,
    ), f"Expected output shape (1, 64, 32, 32), but got {out.shape}"

    resnet_18 = ResNet_ImageNet(ResidualBlock, [2, 2, 2, 2], num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    out = resnet_18(x)
    assert out.shape == (1, 100), f"Expected output shape (1, 100), but got {out.shape}"

    resnet_cifar = ResNet_CIFAR(ResidualBlock, [2, 2, 2], num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    out = resnet_cifar(x)
    assert out.shape == (1, 100), f"Expected output shape (1, 100), but got {out.shape}"

    print("All tests passed!")
