"""
Image transformation functions for M-stream (low-pass grayscale) 
and P-stream (high-pass color).
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter


class LowPassGrayscale:
    """
    Low-pass filter + grayscale for M-stream.
    Simulates magnocellular pathway: global structure, contrast, no fine detail.
    """
    def __init__(self, sigma=3.0):
        self.sigma = sigma
    
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        
        # Apply Gaussian blur to each channel separately
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Color image: blur each channel
            blurred = np.zeros_like(img_np)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=self.sigma)
        else:
            # Grayscale image
            blurred = gaussian_filter(img_np, sigma=self.sigma)
        
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        blurred_pil = Image.fromarray(blurred)
        grayscale = blurred_pil.convert('L')
        return grayscale


class HighPassColor:
    """
    High-pass filter + color preservation for P-stream.
    Simulates parvocellular pathway: fine details, edges, color, textures.
    """
    def __init__(self, sigma=3.0, strength=1.0):
        self.sigma = sigma
        self.strength = strength
    
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        
        # Apply Gaussian blur to each channel separately
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            blurred = np.zeros_like(img_np)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=self.sigma)
        else:
            blurred = gaussian_filter(img_np, sigma=self.sigma)
        
        # High frequencies = original - low frequencies
        high_pass = img_np - blurred
        
        # Amplify high frequencies
        high_pass = high_pass * self.strength
        
        # Add back a small amount of low frequencies to keep image viewable
        result = blurred * 0.3 + high_pass
        
        # Clip and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result.astype('uint8'))


def get_m_stream_transform(sigma=3.0, img_size=128):
    return transforms.Compose([
        LowPassGrayscale(sigma=sigma),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])


def get_p_stream_transform(sigma=3.0, strength=1.0, img_size=128):
    return transforms.Compose([
        HighPassColor(sigma=sigma, strength=strength),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_standard_transform(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
