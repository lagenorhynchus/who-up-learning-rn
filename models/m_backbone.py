import torch
import torch.nn as nn


class MStreamBackbone(nn.Module):
    """
    Lightweight CNN mirroring the magnocellular (M) visual pathway.

    Input  : (B, 1, H, W)  — single-channel blurred grayscale
    Output : (B, 128)       — flat feature vector, no classification head

    Architecture (input-size agnostic via AdaptiveAvgPool2d):

        Stage 0  Conv 1→32,  stride 1
        Stage 1  Conv 32→64, stride 2
        Stage 2  Conv 64→128,stride 2
        Stage 3  Conv 128→128,stride 2
        GAP                             →  (B, 128, 1, 1)
        Flatten                         →  (B, 128)

    Biological grounding:
        Achromatic (1-ch grayscale), low spatial frequency (upstream Gaussian blur),
        large receptive fields (stride-2 downsampling), no FC head — feeds fusion layer.
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # stage 0: 1 → 32, preserve resolution
            nn.Conv2d(1,  32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # stage 1: 32 → 64, halve spatial dims
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # stage 2: 64 → 128, halve spatial dims
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # stage 3: 128 → 128, halve spatial dims
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Collapses any spatial size → (B, 128, 1, 1), making backbone input-size agnostic
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor  shape (B, 128)
        """
        x = self.features(x)
        x = self.gap(x)          # (B, 128, 1, 1)
        x = torch.flatten(x, 1)  # (B, 128)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}\n")

    model = MStreamBackbone().to(device)
    model.eval()

    dummy = torch.randn(8, 1, 128, 128, device=device)

    with torch.no_grad():
        latent = model(dummy)

    print("MStreamBackbone")
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {latent.shape}")
    assert latent.shape == (8, 128), f"Unexpected output shape: {latent.shape}"
    print("  Shape assertion passed.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}")
