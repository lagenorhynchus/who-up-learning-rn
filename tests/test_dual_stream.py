"""Tests for DualStreamCNN. Run from project root: python tests/test_dual_stream.py"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from models import DualStreamCNN


def test_forward_shape_10class():
    """10-class subset: (B, 1, 128, 128) + (B, 3, 128, 128) -> (B, 10) logits."""
    model = DualStreamCNN(num_classes=10)
    m = torch.randn(8, 1, 128, 128)
    p = torch.randn(8, 3, 128, 128)
    out = model(m, p)
    assert tuple(out.shape) == (8, 10), f"got {tuple(out.shape)}"
    print("test_forward_shape_10class passed")


def test_forward_shape_100class():
    """Full CIFAR-100: (B, 1, 128, 128) + (B, 3, 128, 128) -> (B, 100) logits."""
    model = DualStreamCNN(num_classes=100)
    m = torch.randn(4, 1, 128, 128)
    p = torch.randn(4, 3, 128, 128)
    out = model(m, p)
    assert tuple(out.shape) == (4, 100), f"got {tuple(out.shape)}"
    print("test_forward_shape_100class passed")


def test_no_nans():
    """Forward pass on random input produces finite logits."""
    model = DualStreamCNN(num_classes=10)
    out = model(torch.randn(4, 1, 128, 128), torch.randn(4, 3, 128, 128))
    assert not torch.isnan(out).any()
    print("test_no_nans passed")


def test_gradient_flow():
    """Cross-entropy backward produces non-zero gradient on every learnable param."""
    model = DualStreamCNN(num_classes=10)
    m = torch.randn(4, 1, 128, 128)
    p = torch.randn(4, 3, 128, 128)
    logits = model(m, p)
    target = torch.randint(0, 10, (4,))
    loss = F.cross_entropy(logits, target)
    loss.backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum().item() > 0, f"{name} has zero grad"
    print("test_gradient_flow passed")


def test_alpha_fixed_is_buffer():
    """learnable_alpha=False: alpha is a buffer (no grad) and equals the constructor value."""
    model = DualStreamCNN(num_classes=10, alpha=0.7, learnable_alpha=False)
    assert "_alpha_raw" not in dict(model.named_parameters())
    np.testing.assert_allclose(model.alpha.item(), 0.7, atol=1e-6)
    print("test_alpha_fixed_is_buffer passed")


def test_alpha_learnable_receives_gradient():
    """learnable_alpha=True: alpha is a Parameter and gets a non-zero gradient from loss.backward()."""
    model = DualStreamCNN(num_classes=10, alpha=0.5, learnable_alpha=True)
    assert "_alpha_raw" in dict(model.named_parameters())
    logits = model(torch.randn(4, 1, 128, 128), torch.randn(4, 3, 128, 128))
    target = torch.randint(0, 10, (4,))
    F.cross_entropy(logits, target).backward()
    assert model._alpha_raw.grad is not None
    assert model._alpha_raw.grad.abs().item() > 0, "alpha received no gradient"
    print("test_alpha_learnable_receives_gradient passed")


def main():
    test_forward_shape_10class()
    test_forward_shape_100class()
    test_no_nans()
    test_gradient_flow()
    test_alpha_fixed_is_buffer()
    test_alpha_learnable_receives_gradient()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
