import torch
import torch.nn as nn


def center_crop_2d(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    if target_h > h or target_w > w:
        raise ValueError(
            f"Cannot center-crop tensor of shape {(h, w)} to larger target {(target_h, target_w)}"
        )
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[:, :, top : top + target_h, left : left + target_w]


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = center_crop_2d(skip, x.shape[2], x.shape[3])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.final_conv(x)

    def trace_shapes(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[Trace] input:           {tuple(x.shape)}")

        skip1, x = self.down1(x)
        print(f"[Trace] enc1:            {tuple(skip1.shape)}")
        print(f"[Trace] pool1:           {tuple(x.shape)}")

        skip2, x = self.down2(x)
        print(f"[Trace] enc2:            {tuple(skip2.shape)}")
        print(f"[Trace] pool2:           {tuple(x.shape)}")

        skip3, x = self.down3(x)
        print(f"[Trace] enc3:            {tuple(skip3.shape)}")
        print(f"[Trace] pool3:           {tuple(x.shape)}")

        skip4, x = self.down4(x)
        print(f"[Trace] enc4:            {tuple(skip4.shape)}")
        print(f"[Trace] pool4:           {tuple(x.shape)}")

        x = self.bottleneck(x)
        print(f"[Trace] bottleneck:      {tuple(x.shape)}")

        x = self.up1.up(x)
        print(f"[Trace] up1 transposed:  {tuple(x.shape)}")
        s4 = center_crop_2d(skip4, x.shape[2], x.shape[3])
        x = torch.cat([s4, x], dim=1)
        print(f"[Trace] up1 concat:      {tuple(x.shape)}")
        x = self.up1.double_conv(x)
        print(f"[Trace] up1 out:         {tuple(x.shape)}")

        x = self.up2.up(x)
        print(f"[Trace] up2 transposed:  {tuple(x.shape)}")
        s3 = center_crop_2d(skip3, x.shape[2], x.shape[3])
        x = torch.cat([s3, x], dim=1)
        print(f"[Trace] up2 concat:      {tuple(x.shape)}")
        x = self.up2.double_conv(x)
        print(f"[Trace] up2 out:         {tuple(x.shape)}")

        x = self.up3.up(x)
        print(f"[Trace] up3 transposed:  {tuple(x.shape)}")
        s2 = center_crop_2d(skip2, x.shape[2], x.shape[3])
        x = torch.cat([s2, x], dim=1)
        print(f"[Trace] up3 concat:      {tuple(x.shape)}")
        x = self.up3.double_conv(x)
        print(f"[Trace] up3 out:         {tuple(x.shape)}")

        x = self.up4.up(x)
        print(f"[Trace] up4 transposed:  {tuple(x.shape)}")
        s1 = center_crop_2d(skip1, x.shape[2], x.shape[3])
        x = torch.cat([s1, x], dim=1)
        print(f"[Trace] up4 concat:      {tuple(x.shape)}")
        x = self.up4.double_conv(x)
        print(f"[Trace] up4 out:         {tuple(x.shape)}")

        x = self.final_conv(x)
        print(f"[Trace] final output:    {tuple(x.shape)}")
        return x


def validate_unet_input_size(image_size: int) -> None:
    """
    Validate input size for valid-convolution U-Net.

    Applies the recurrence relation for each encoder stage:
        s_{i+1} = (s_i - 4) / 2
    
    where two 3x3 valid convs reduce by 4, and 2x2 maxpool halves the result.
    
    All intermediate sizes must be positive integers.
    Bottleneck must also satisfy: s_bottleneck = s_4 - 4 > 0
    
    Args:
        image_size: Input spatial dimension (assumed square).
        
    Raises:
        ValueError: If any constraint is violated.
    """
    s = image_size
    
    # Encoder stages: 4 down blocks
    sizes = [s]
    for level in range(1, 5):
        # Two valid convs reduce by 4, then maxpool halves
        if (s - 4) % 2 != 0:
            raise ValueError(
                f"At encoder level {level}: size {s} → ({s} - 4) / 2 is not an integer. "
                f"({s} - 4) = {s - 4} must be even."
            )
        s = (s - 4) // 2
        if s <= 0:
            raise ValueError(
                f"At encoder level {level}: size becomes non-positive ({s}). "
                f"Input size {image_size} is too small."
            )
        sizes.append(s)
    
    # Bottleneck: two more valid convs without pooling
    s_bottleneck = s - 4
    if s_bottleneck <= 0:
        raise ValueError(
            f"Bottleneck size {s_bottleneck} is non-positive. "
            f"Input size {image_size} is too small."
        )
    
    # All constraints passed
    print(
        f"[Input validation] size {image_size} is valid for valid-conv U-Net. "
        f"Encoder path: {' → '.join(map(str, sizes))} → bottleneck {s_bottleneck}"
    )