import torch.nn as nn
from torch import cat

# written by David

# the network structure of U-Net is:
# during encoding alternating blocks of convolutions and max pool operations
# during decoding alternating blocks of convolutions and up convolutions
# the blocks of convolutions consist of one conv, one relu, one conv and one relu
# during decoding additionally activations of former layers are copied, cropped and concatenated
# to the activations of a current layer


def double_convolution_without_bn(in_channels, out_channels):
    kernel_size = 3
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size),
        nn.ReLU(True))


def double_convolution_bn(in_channels, out_channels):
    kernel_size = 3
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True))


def double_convolution(in_channels, out_channels, batch_norm):
    if batch_norm:
        return double_convolution_bn(in_channels, out_channels)
    else:
        return double_convolution_without_bn(in_channels, out_channels)


def single_convolution(in_channels, out_channels):
    kernel_size = 3
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.ReLU(True))


def convolution(in_channels, out_channels, halve, batch_norm):
    if halve:
        return single_convolution(in_channels, out_channels)
    else:
        return double_convolution(in_channels, out_channels, batch_norm)

def copy_crop_and_concat(tensor_1, tensor_2):
    tensor_1_size = tensor_1.size()[2]
    tensor_2_size = tensor_2.size()[2]
    edge = (tensor_1_size - tensor_2_size) // 2  # calculates the edge
    cropped = tensor_1[:, :, edge:tensor_1_size - edge, edge:tensor_1_size - edge]  # crops the copy
    return cat([cropped, tensor_2], 1)  # concatenates the both tensors


class UNet(nn.Module):
    def __init__(self, class_count, color_channels=3, halve=False, batch_norm=False):
        super(UNet, self).__init__()

        kernel_size = 2
        stride = 2

        # encoder convolutions
        self.conv_1 = convolution(color_channels, 64, halve, batch_norm)
        self.conv_2 = convolution(64, 128, halve, batch_norm)
        self.conv_3 = convolution(128, 256, halve, batch_norm)
        self.conv_4 = convolution(256, 512, halve, batch_norm)
        self.conv_5 = convolution(512, 1024, halve, batch_norm)

        # maxpooling
        self.max_pool = nn.MaxPool2d(kernel_size, stride)

        # decoder convolutions
        self.conv_6 = convolution(1024, 512, halve, batch_norm)
        self.conv_7 = convolution(512, 256, halve, batch_norm)
        self.conv_8 = convolution(256, 128, halve, batch_norm)
        self.conv_9 = convolution(128, 64, halve, batch_norm)

        # upconvolutions
        self.up_conv_1 = nn.ConvTranspose2d(1024, 512, kernel_size, stride)
        self.up_conv_2 = nn.ConvTranspose2d(512, 256, kernel_size, stride)
        self.up_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size, stride)
        self.up_conv_4 = nn.ConvTranspose2d(128, 64, kernel_size, stride)

        # output layer
        self.out = nn.Conv2d(64, class_count, kernel_size=1)

    def forward(self, x):
        # encoder
        conv_1 = self.conv_1(x)
        max_pool_1 = self.max_pool(conv_1)
        conv_2 = self.conv_2(max_pool_1)
        max_pool_2 = self.max_pool(conv_2)
        conv_3 = self.conv_3(max_pool_2)
        max_pool_3 = self.max_pool(conv_3)
        conv_4 = self.conv_4(max_pool_3)
        max_pool_4 = self.max_pool(conv_4)
        conv_5 = self.conv_5(max_pool_4)

        # decoder
        x = self.up_conv_1(conv_5)
        x = self.conv_6(copy_crop_and_concat(conv_4, x))
        x = self.up_conv_2(x)
        x = self.conv_7(copy_crop_and_concat(conv_3, x))
        x = self.up_conv_3(x)
        x = self.conv_8(copy_crop_and_concat(conv_2, x))
        x = self.up_conv_4(x)
        x = self.conv_9(copy_crop_and_concat(conv_1, x))

        # output
        x = self.out(x)
        return x
