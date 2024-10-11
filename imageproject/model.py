import torch
import torch.nn as nn
import torchvision.models as models
from torch.fft import fft2, ifft2

# Frequency-based Feature Aggregation (FFA) module
class FFAModule(nn.Module):
    def __init__(self, channels):
        super(FFAModule, self).__init__()
        self.conv1 = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, x_prev, x_next):
        # Apply FFT to the inputs
        f_x = fft2(x)
        f_x_prev = fft2(x_prev)
        f_x_next = fft2(x_next)

        # Concatenate real and imaginary parts
        f_x_combined = torch.cat([torch.real(f_x), torch.imag(f_x)], dim=1)
        f_x_prev_combined = torch.cat([torch.real(f_x_prev), torch.imag(f_x_prev)], dim=1)
        f_x_next_combined = torch.cat([torch.real(f_x_next), torch.imag(f_x_next)], dim=1)

        # Frequency aggregation with channel attention
        y1 = self.conv1(f_x_combined) * f_x_combined
        y2 = self.conv2(f_x_prev_combined) * f_x_next_combined

        # Inverse FFT to return to spatial domain
        y1 = torch.fft.ifft2(y1)
        y2 = torch.fft.ifft2(y2)

        return y1 + y2

# Two-branch Decoder for segmentation and localization
class TwoBranchDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoBranchDecoder, self).__init__()
        self.segmentation_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        self.localization_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        seg_output = self.segmentation_branch(x)
        loc_output = self.localization_branch(x)
        return seg_output, loc_output

# FLA-Net Model Definition
class FLANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(FLANet, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove the final layer

        # Frequency feature aggregation
        self.ffa = FFAModule(2048)  # 2048 comes from the final layer of ResNet50

        # Two-branch decoder
        self.decoder = TwoBranchDecoder(2048, out_channels)

    def forward(self, x, x_prev, x_next):
        # Encode the frames
        enc_x = self.encoder(x)
        enc_x_prev = self.encoder(x_prev)
        enc_x_next = self.encoder(x_next)

        # Apply Frequency-based Feature Aggregation
        ffa_output = self.ffa(enc_x, enc_x_prev, enc_x_next)

        # Two-branch decoding
        segmentation, localization = self.decoder(ffa_output)

        return segmentation, localization
