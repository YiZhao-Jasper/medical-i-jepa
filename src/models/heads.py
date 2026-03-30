import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    """Linear probe head: global average pooling + linear layer."""

    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        x = tokens.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


class AttentiveClassifier(nn.Module):
    """Attentive pooling + MLP classifier for richer feature aggregation."""

    def __init__(self, embed_dim, num_classes, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, tokens):
        B = tokens.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, tokens, tokens)
        out = self.norm(out.squeeze(1))
        return self.mlp(out)


class SegmentationDecoder(nn.Module):
    """Lightweight convolutional decoder for ViT patch tokens → segmentation mask."""

    def __init__(self, embed_dim, num_classes=1, channels=None, img_size=224):
        super().__init__()
        if channels is None:
            channels = [256, 128, 64]

        self.img_size = img_size

        layers = []
        in_ch = embed_dim
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2),
            ])
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, num_classes, 1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, tokens, grid_size):
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return x
