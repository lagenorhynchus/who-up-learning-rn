import torch
import torch.nn as nn

class TextureStream(nn.Module):
    """
    Lightweight CNN that focuses on texture features.

    Input  : (B, 1, H, W)  — single-channel blurred grayscale
    Output : (B, latent_dim)  — flat feature vector, no classification head

    Architecture (designed for 32×32 CIFAR inputs):

        Stage 0  Conv 1→32,  stride 1  →  (B, 32, 32, 32)
        Stage 1  Conv 32→64, stride 2  →  (B, 64, 16, 16)
        Stage 2  Conv 64→128,stride 2  →  (B,128,  8,  8)
        Stage 3  Conv 128→128,stride 2 →  (B,128,  4,  4)
        GAP                             →  (B,128,  1,  1)
        Flatten + Linear                →  (B, latent_dim)

    Visual cells in the magnocellular pathway in the brain encode coarse,
    low-frequency structure
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.features = nn.Sequential(
            # stage 0: 1 → 32, preserve resolution
            nn.Conv2d(1,  32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # stage 1: 32 → 64, 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # stage 2: 64 → 128, 16 → 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # stage 3: 128 → 128, 8 → 4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling collapses spatial dims → (B, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Linear projection to latent space (no activation — caller decides)
        self.projector = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor  shape (B, latent_dim)
        """
        x = self.features(x)        # (B, 128, 4, 4) for 32×32 input
        x = self.gap(x)             # (B, 128, 1, 1)
        x = torch.flatten(x, 1)    # (B, 128)
        x = self.projector(x)       # (B, latent_dim)
        return x


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}\n")

    model = TextureStream(latent_dim=256).to(device)
    model.eval()

    # Dummy tensor that matches MagnocellularTransform output: (B, 1, 32, 32)
    dummy = torch.randn(8, 1, 32, 32, device=device)

    with torch.no_grad():
        latent = model(dummy)

    print("TextureStream")
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {latent.shape}")   # expect (8, 256)
    assert latent.shape == (8, 256), f"Unexpected output shape: {latent.shape}"
    print("  Shape assertion passed.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}")
