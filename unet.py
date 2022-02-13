import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1, bias=False),  # Note:in_channels = out_ch
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ups = nn.ModuleList()

        # encoder
        for feature in features:
            self.downs.append(ConvBlock(in_ch, feature))
            in_ch = feature

        # bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):

        # contraction path
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bridge 
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # expansion path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return x


class UNetRefiner(nn.Module):
    def __init__(self, model, by_freezing=True):
        super(UNetRefiner, self).__init__()

        self.downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if by_freezing:
            for param in model.parameters():
                param.requires_grad = False
        self.model = model

        self.upsample = nn.Upsample(scale_factor=2)
        self.refine1 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), padding=1, stride=1)
        self.refine2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=1, stride=1)

    def forward(self, x):
        img_high_resolution = x[:, 0:3, :, :]
        x = self.downsample(x)
        x = self.model(x)
        x = self.upsample(x)

        # refiner
        x = torch.cat((img_high_resolution, x), dim=1)
        x = self.refine1(x)
        x = self.refine2(x)

        return x


def test():
    x = torch.randn(8, 9, 512, 512)

    # frozen model
    unet = UNet(in_ch=9, out_ch=1)
    frozen_unet_refiner = UNetRefiner(model=unet, by_freezing=True)
    assert list(frozen_unet_refiner.model.downs.parameters())[0].requires_grad is False

    # trainable model
    unet = UNet(in_ch=9, out_ch=1)
    trainable_unet_refiner = UNetRefiner(model=unet, by_freezing=False)
    assert list(trainable_unet_refiner.model.downs.parameters())[0].requires_grad is True

    pred = frozen_unet_refiner(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", pred.shape)

    assert x[:, 0, :, :].shape == pred[:, 0, :, :].shape


if __name__ == '__main__':
    test()
