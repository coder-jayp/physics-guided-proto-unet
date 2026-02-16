import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet (Decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom of the U
        x = self.bottleneck(x)
        
        # Reverse skips for the decoder
        skip_connections = skip_connections[::-1]

        # Decoder path
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # ConvTranspose2d
            skip_connection = skip_connections[i//2]

            # Handle non-power-of-2 dimensions (like your 600x600)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip) # DoubleConv

        return self.final_conv(x)

def test():
    """Unit test to verify shape consistency"""
    # Simulate a batch of 3 noisy disk images
    x = torch.randn((3, 1, 600, 600))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    
    print(f"--- UNet Architecture Test ---")
    print(f"Input Tensor Shape:  {x.shape}")
    print(f"Output Tensor Shape: {preds.shape}")
    
    assert preds.shape == x.shape, "Error: Output shape does not match input!"
    print("Test Passed: Dimensions are consistent.")

if __name__ == "__main__":
    test()