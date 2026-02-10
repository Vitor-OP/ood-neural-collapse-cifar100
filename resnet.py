"""
In this file we implement the ResNet architecture described in https://arxiv.org/pdf/1512.03385

Layers:
Conv Input (stride 2, 64 feature maps) - Halves the image (224 -> 112)
Max Pool (stride 2) - Halves the image (112 -> 56)
Stack 1 (no downsampling, 64 feature maps) - Image size unchanged (56 -> 56)
Stack 2 (downsampling at start, 128 feature maps) - Halves the image (56 -> 28)
Stack 3 (downsampling at start, 256 feature maps) - Halves the image (28 -> 14)
Stack 4 (downsampling at start, 512 feature maps) - Halves the image (14 -> 7)
Average Pool - Reduces each feature map to a single number (7 -> 1)
Fully Connected Output (n classes) - Reduces to n classes (512 -> n)
Softmax - Converts to n probabilities

The paper describes this architecture and the convolutions sizes for ImageNet.

The paper also mentions training in CIFAR-10, and gives a simpler architecture, with only 3 stacks,
with 2 blocks per stack, and feature maps 16, 32, and 64. It also changes the initial convolution
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