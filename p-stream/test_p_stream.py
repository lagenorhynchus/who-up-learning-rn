import torch
import numpy as np

from p_transform import PStreamTransform
from p_backbone import PStreamBackbone


def test_transform_preserves_shape():
    """Transform returns same shape as input for a (B, 3, H, W) batch."""
    t = PStreamTransform()
    out = t(torch.randn(16, 3, 32, 32))
    assert tuple(out.shape) == (16, 3, 32, 32), f"got {tuple(out.shape)}"
    print("test_transform_preserves_shape passed")


def test_transform_zero_on_flat_input():
    """A constant (flat) image has no high-frequency content, so output is ~0."""
    t = PStreamTransform()
    flat = torch.ones(1, 3, 32, 32) * 0.5
    out = t(flat)
    np.testing.assert_array_less(out.abs().max().item(), 0.01)
    print("test_transform_zero_on_flat_input passed")


def test_backbone_output_shape():
    """Backbone maps (B, 3, 32, 32) to a (B, 128) feature vector."""
    model = PStreamBackbone()
    out = model(torch.randn(16, 3, 32, 32))
    assert tuple(out.shape) == (16, 128), f"got {tuple(out.shape)}"
    print("test_backbone_output_shape passed")


def test_full_pipeline_no_nans():
    """Transform then backbone on random input yields a finite (B, 128) feature vector."""
    t = PStreamTransform()
    model = PStreamBackbone()
    out = model(t(torch.randn(16, 3, 32, 32)))
    assert tuple(out.shape) == (16, 128)
    assert not torch.isnan(out).any()
    print("test_full_pipeline_no_nans passed")


def test_gradient_flow():
    """Backprop through transform + backbone produces a non-zero gradient on every learnable param."""
    t = PStreamTransform()
    model = PStreamBackbone()
    out = model(t(torch.randn(16, 3, 32, 32)))
    out.sum().backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum().item() > 0, f"{name} has zero grad"
    print("test_gradient_flow passed")


def main():
    test_transform_preserves_shape()
    test_transform_zero_on_flat_input()
    test_backbone_output_shape()
    test_full_pipeline_no_nans()
    test_gradient_flow()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
