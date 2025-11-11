import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (to be upsampled)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for music source separation.
    Input: Magnitude spectrogram of mixed audio (batch, channels, freq, time)
    Output: Magnitude spectrograms for each stem (batch, num_stems * channels, freq, time)
    """
    
    def __init__(self, 
                 in_channels=2, 
                 out_channels=2,
                 num_stems=4,
                 base_channels=32,
                 depth=6,
                 kernel_size=5):
        """
        Args:
            in_channels: Number of input channels (2 for stereo)
            out_channels: Number of output channels per stem
            num_stems: Number of stems to separate
            base_channels: Number of channels in first layer
            depth: Depth of U-Net (number of down/up blocks)
            kernel_size: Kernel size for convolutions
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stems = num_stems
        self.depth = depth
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels, kernel_size, padding)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(depth):
            next_channels = current_channels * 2
            self.down_blocks.append(Down(current_channels, next_channels, kernel_size))
            current_channels = next_channels
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            next_channels = current_channels // 2
            self.up_blocks.append(Up(current_channels, next_channels, kernel_size))
            current_channels = next_channels
        
        # Output convolution for each stem
        self.outc = nn.Conv2d(base_channels, num_stems * out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: Input magnitude spectrogram (batch, channels, freq, time)
            
        Returns:
            Output magnitude spectrograms for all stems (batch, num_stems * channels, freq, time)
        """
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.inc(x)
        skip_connections.append(x)
        
        # Encoder
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (it's the bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder with skip connections
        for up in self.up_blocks:
            skip = skip_connections.pop()
            x = up(x, skip)
        
        # Output layer
        x = self.outc(x)
        
        return x
    
    def separate_stems(self, x):
        """
        Forward pass that returns separated stems as a dictionary.
        
        Args:
            x: Input magnitude spectrogram (batch, channels, freq, time)
            
        Returns:
            Dictionary of stem spectrograms {stem_idx: (batch, channels, freq, time)}
        """
        output = self.forward(x)
        batch_size, _, freq, time = output.shape
        
        # Reshape to separate stems
        # From (batch, num_stems * channels, freq, time)
        # To (batch, num_stems, channels, freq, time)
        output = output.view(batch_size, self.num_stems, self.out_channels, freq, time)
        
        # Return as dictionary
        stems = {}
        for i in range(self.num_stems):
            stems[i] = output[:, i, :, :, :]
        
        return stems


class MaskBasedUNet(UNet):
    """
    U-Net that predicts masks instead of direct magnitude spectrograms.
    Masks are multiplied with input to get separated sources.
    This often works better for source separation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Output activation to ensure masks are in [0, 1]
        self.mask_activation = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input magnitude spectrogram (batch, channels, freq, time)
            
        Returns:
            Separated magnitude spectrograms (batch, num_stems * channels, freq, time)
        """
        # Get masks from U-Net
        masks = super().forward(x)
        masks = self.mask_activation(masks)
        
        # Reshape masks
        batch_size, _, freq, time = masks.shape
        masks = masks.view(batch_size, self.num_stems, self.out_channels, freq, time)
        
        # Apply masks to input
        x_expanded = x.unsqueeze(1)  # (batch, 1, channels, freq, time)
        separated = masks * x_expanded  # (batch, num_stems, channels, freq, time)
        
        # Reshape back
        separated = separated.view(batch_size, self.num_stems * self.out_channels, freq, time)
        
        return separated
    
    def separate_stems(self, x):
        """
        Forward pass that returns separated stems as a dictionary.
        
        Args:
            x: Input magnitude spectrogram (batch, channels, freq, time)
            
        Returns:
            Dictionary of stem spectrograms {stem_idx: (batch, channels, freq, time)}
        """
        output = self.forward(x)
        batch_size, _, freq, time = output.shape
        
        # Reshape to separate stems
        output = output.view(batch_size, self.num_stems, self.out_channels, freq, time)
        
        # Return as dictionary
        stems = {}
        for i in range(self.num_stems):
            stems[i] = output[:, i, :, :, :]
        
        return stems


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input (batch=2, channels=2, freq=1025, time=256)
    x = torch.randn(2, 2, 1025, 256).to(device)
    
    print("Testing UNet...")
    model = UNet(in_channels=2, out_channels=2, num_stems=4, base_channels=32, depth=4).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    stems = model.separate_stems(x)
    print(f"Separated stems: {len(stems)}")
    for i, stem in stems.items():
        print(f"  Stem {i} shape: {stem.shape}")
    
    print("\nTesting MaskBasedUNet...")
    mask_model = MaskBasedUNet(in_channels=2, out_channels=2, num_stems=4, base_channels=32, depth=4).to(device)
    print(f"Parameters: {count_parameters(mask_model):,}")
    
    output = mask_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    stems = mask_model.separate_stems(x)
    print(f"Separated stems: {len(stems)}")
    for i, stem in stems.items():
        print(f"  Stem {i} shape: {stem.shape}")