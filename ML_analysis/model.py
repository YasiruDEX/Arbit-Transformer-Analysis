import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAutoEncoder(nn.Module):
    """AutoEncoder for thermal image anomaly detection"""
    
    def __init__(self, in_channels=3, latent_dim=128):
        super(AnomalyAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, x):
        """Forward pass through encoder-decoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def get_anomaly_map(self, x):
        """
        Generate pixel-wise anomaly map
        Higher values indicate more anomalous regions
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Calculate reconstruction error per pixel
            anomaly_map = torch.abs(x - reconstructed)
            # Average across channels to get single-channel map
            anomaly_map = torch.mean(anomaly_map, dim=1, keepdim=True)
        return anomaly_map, reconstructed


class ImprovedAutoEncoder(nn.Module):
    """Improved AutoEncoder with skip connections"""
    
    def __init__(self, in_channels=3):
        super(ImprovedAutoEncoder, self).__init__()
        
        # Encoder blocks
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)  # 512 because of skip connection
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        self.out = nn.Conv2d(32, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        return self.sigmoid(out)
    
    def get_anomaly_map(self, x):
        """Generate pixel-wise anomaly map"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            anomaly_map = torch.abs(x - reconstructed)
            anomaly_map = torch.mean(anomaly_map, dim=1, keepdim=True)
        return anomaly_map, reconstructed
