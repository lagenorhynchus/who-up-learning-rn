import torch
import torch.nn as nn

from .m_backbone import MStreamBackbone
from .p_backbone import PStreamBackbone


class DualStreamCNN(nn.Module):
    """
    Dual-stream model combining the magnocellular (M) and parvocellular (P) pathways.

    Forward pass:
        m_features = MStreamBackbone(m_input)   # (B, 128)
        p_features = PStreamBackbone(p_input)   # (B, 128)
        fused      = alpha * m_features + (1 - alpha) * p_features  # (B, 128)
        logits     = classifier(fused)           # (B, num_classes)

    Alpha controls the M/P balance:
        alpha=1.0  → pure magnocellular (texture/low-freq)
        alpha=0.0  → pure parvocellular (shape/high-freq)
        alpha=0.5  → equal blend (default)

    Alpha can be a fixed float (set at construction) or a learnable parameter
    (pass learnable_alpha=True) to let the model discover the optimal blend.
    """

    def __init__(
        self,
        num_classes: int = 10,
        alpha: float = 0.5,
        learnable_alpha: bool = False,
    ) -> None:
        super().__init__()

        self.m_backbone = MStreamBackbone()   # accepts (B, 1, H, W)
        self.p_backbone = PStreamBackbone()   # accepts (B, 3, H, W)

        if learnable_alpha:
            # Unconstrained parameter; sigmoid at forward time keeps it in (0, 1)
            self._alpha_raw = nn.Parameter(torch.tensor(alpha))
            self.learnable_alpha = True
        else:
            self.register_buffer("_alpha_raw", torch.tensor(alpha))
            self.learnable_alpha = False

        # Maps 128-dim fused vector to class logits
        self.classifier = nn.Linear(128, num_classes)

    @property
    def alpha(self) -> torch.Tensor:
        # Constrain to (0, 1) regardless of whether alpha is learned or fixed
        return torch.sigmoid(self._alpha_raw) if self.learnable_alpha else self._alpha_raw

    def forward(
        self, m_input: torch.Tensor, p_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        m_input : torch.Tensor  shape (B, 1, H, W)  — M-stream (grayscale low-pass)
        p_input : torch.Tensor  shape (B, 3, H, W)  — P-stream (RGB high-pass)

        Returns
        -------
        torch.Tensor  shape (B, num_classes) — raw logits (no softmax)
        """
        m_feat = self.m_backbone(m_input)   # (B, 128)
        p_feat = self.p_backbone(p_input)   # (B, 128)

        alpha = self.alpha
        fused = alpha * m_feat + (1.0 - alpha) * p_feat  # (B, 128)

        return self.classifier(fused)        # (B, num_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}\n")

    model = DualStreamCNN(num_classes=10, alpha=0.5).to(device)
    model.eval()

    m_dummy = torch.randn(8, 1, 128, 128, device=device)
    p_dummy = torch.randn(8, 3, 128, 128, device=device)

    with torch.no_grad():
        logits = model(m_dummy, p_dummy)

    print("DualStreamCNN")
    print(f"  M input  shape : {m_dummy.shape}")
    print(f"  P input  shape : {p_dummy.shape}")
    print(f"  Output   shape : {logits.shape}")
    assert logits.shape == (8, 10), f"Unexpected output shape: {logits.shape}"
    print("  Shape assertion passed.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params : {total_params:,}")
    print(f"  Alpha        : {model.alpha.item():.3f}")
